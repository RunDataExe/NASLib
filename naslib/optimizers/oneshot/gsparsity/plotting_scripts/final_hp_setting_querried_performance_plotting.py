#
#! Hyperparameter wide, narrow, final setting runs


#! Input
#  Should take folder as input and plot results per dataset across all methods and seeds
# also once show accross datasets as some sort of generalization plot

# TODO this is currently only for final hpo setting I will have to come up with how to do this for HPO wide and narrow and maybe also how to visualize HP Importance in each setting
# TODO AUC Number calculation

# * Time vs Performance
# Area under Curve Plots
# Each with RAW times and once with times that are normalized into [0,1] range
# Plot once for train_acc and once for val_acc per dataset
# Plot once for train_acc and once for val_acc across datasets
#! One Stage
# ? Time
# Runtime
# ? Accuracy (train/val)
# train_acc, val_acc

#! Two Stage
# ? Time
# For first "epoch" (e.g. all epochs of stage1 and first epoch of stage2): SUM(Querried Training Times that incoporates computational factor per dataset) * 200 + first runtime entry
# For following "epochs" (e.g. other epochs of stage2): next runtime entries
# ? Accuracy (train/val)
# train_acc, val_acc corresponding to the runtime entries (neglect corresponding to train_time)

#! Random Sampling (random search with one epoch) / Random Search
# ? Time
# Querried Training Time that incoporates computational factor per dataset * 200
# ? Accuracy (train/val)
# train_acc, val_acc

# * Time vs Stability
# Area between between Seeds of a Method Plots
# Each with RAW times and once with times that are normalized into [0,1] range
# Plot once for train_acc and once for val_acc per dataset
# Plot once for train_acc and once for val_acc across datasets

#! Use the same way of getting Time and Accuracy as above

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# from sklearn.metrics import auc # No longer needed
import argparse
import re
from collections import defaultdict
import matplotlib.colors as mcolors

# Dataset classes for random guess calculation
DATASET_CLASSES = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}

# Default color and style settings for plots
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
# New color palette provided by the user, converted from [0, 255] to [0, 1] range
DEFAULTS = [
    (0.5490196078431373, 0.19215686274509805, 1.0),
    (0.20392156862745098, 0.8901960784313725, 0.3411764705882353),
    (0.9764705882352941, 0.0, 0.6313725490196078),
    (0.1803921568627451, 0.32941176470588235, 0.0),
    (0.00392156862745098, 0.29411764705882354, 0.6666666666666666),
    (0.7843137254901961, 0.8, 0.44313725490196076),
    (1.0, 0.5137254901960784, 0.34509803921568627),
    (0.00392156862745098, 0.5686274509803921, 0.5254901960784314),
    (0.5568627450980392, 0.43137254901960786, 0.0),
    (0.9725490196078431, 0.7254901960784313, 0.5607843137254902),
]
C_MAX = 10
COLORS = [*DEFAULTS[:C_MAX]] * 3
FMTS = [*["-"] * C_MAX, *["--"] * C_MAX, *[":"] * C_MAX]
# Prioritize the most visually distinct markers for the first few seeds.
# Circle, Square, Plus, Diamond, and Cross are highly discriminable.
MARKERS = ["o", "s", "+", "D", "x", "^", "*", "v", "<", ">", "p", "h", "H", "P"]

# New helper that draws nicer, compact markers (deterministic jitter, thin contrasting edge, rasterized)
def plot_scatter_markers(
    ax,
    x,
    y,
    color,
    marker,
    seed,
    max_time,
    jitter_frac=1e-3,
    size=30,
    alpha=0.85,
    edgecolor="white",
    linewidth=0.45,
    rasterize=True,
):
    """
    Draw compact markers with:
    - deterministic jitter based on seed to break exact overlap
    - thin contrasting edgecolor to make small markers readable
    - rasterization for dense plots
    jitter_frac: fraction of max_time used as jitter amplitude
    """
    if len(x) == 0:
        return
    try:
        seed_int = int(seed) if isinstance(seed, (int, str)) and str(seed).isdigit() else hash(seed) & 0xFFFFFFFF
    except Exception:
        seed_int = 0
    prng = np.random.RandomState(seed_int)
    jitter_amount = jitter_frac * (max_time if max_time > 0 else 1.0)
    # keep jitter tiny, centered around 0
    jitter = (prng.rand(len(x)) - 0.5) * jitter_amount
    x_j = np.array(x) + jitter

    # Use scatter so we can control face/edge colors and rasterization
    ax.scatter(
        x_j,
        y,
        s=size,
        c=[color],
        marker=marker,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        rasterized=rasterize,
        zorder=6,
    )

def calculate_auc(x, y):
    """
    Calculates the Area Under the Curve (AUC) for a set of (x, y) points.
    This implementation handles non-monotonic x-values by sorting them for calculation,
    which is necessary for a valid geometric interpretation of AUC.
    The y-values are the incumbent (best-so-far) accuracies.
    """
    if len(x) < 2:
        return 0.0

    # Sort points by x-value to apply the trapezoidal rule correctly.
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Ensure y-values are incumbent after sorting by time
    y_incumbent = np.maximum.accumulate(y_sorted)

    # Use the trapezoidal rule for integration.
    # np.trapz is equivalent to sklearn.metrics.auc for sorted inputs.
    return np.trapz(y_incumbent, x_sorted)


def calculate_normalized_auc(x, y, max_time):
    """
    Calculates the Area Under the Curve (AUC) normalized by a maximum time.
    The time axis (x) is scaled to [0, 1] before AUC calculation.
    """
    if len(x) < 2 or max_time <= 0:
        return 0.0

    # Normalize the time axis
    x_normalized = np.array(x) / max_time

    # Sort points by normalized x-value
    sorted_indices = np.argsort(x_normalized)
    x_norm_sorted = x_normalized[sorted_indices]
    y_sorted = np.array(y)[sorted_indices]

    # Ensure y-values are incumbent after sorting by time
    y_incumbent = np.maximum.accumulate(y_sorted)

    # Use the trapezoidal rule on the normalized time axis.
    return np.trapz(y_incumbent, x_norm_sorted)


def get_color_shades(base_color, n_shades):
    """Generates a list of color shades from a base color."""
    if n_shades <= 1:
        return [base_color]
    # Create a color ramp from a slightly lighter version of the base color to the base color
    # This makes the shades distinct but visually related.
    lighter_color = mcolors.to_rgba(base_color, alpha=0.4)
    base_rgba = mcolors.to_rgba(base_color, alpha=0.8)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", [lighter_color, base_rgba]
    )
    return (
        [cmap(i / (n_shades - 1)) for i in range(n_shades)]
        if n_shades > 1
        else [base_rgba]
    )


def format_method_label(optimizer, zcp_method):
    """Human-friendly label for legend."""
    return f"{optimizer} ({zcp_method})" if zcp_method else f"{optimizer}"


def find_error_files(root_dir):
    """Finds all 'errors.json' files and extracts metadata from their paths."""
    error_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "errors.json" in filenames:
            full_path = os.path.join(dirpath, "errors.json")
            # Normalize path for consistent parsing
            relative_path = os.path.relpath(full_path, root_dir)
            parts = relative_path.split(os.sep)

            # Expected structures (relative to root_dir):
            # - With ZCP method: {optimizer}/{zcp_method}/{search_space}/{dataset}/{seed}/errors.json  -> len(parts) == 6
            # - Without ZCP:     {optimizer}/{search_space}/{dataset}/{seed}/errors.json            -> len(parts) == 5
            if len(parts) >= 5:
                optimizer = parts[0]
                seed = parts[-2]
                dataset = parts[-3]
                search_space = parts[-4]
                # Detect zcp_method when there is an extra component
                zcp_method = parts[1] if len(parts) >= 6 else None

                metadata = {
                    "path": full_path,
                    "optimizer": optimizer,
                    "zcp_method": zcp_method,
                    "search_space": search_space,
                    "dataset": dataset,
                    "seed": seed,
                }
                error_files.append(metadata)
    return error_files


def process_run_data(filepath, acc_metric, dataset):
    """
    Loads and processes a single run from an errors.json file.
    Calculates anytime performance based on the cost of search + evaluation.
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Use queried architecture performance for a fair comparison
    queried_acc_metric = (
        "queried_val_acc" if acc_metric == "valid_acc" else "queried_train_acc"
    )

    # Check for required data for a valid anytime performance curve
    if not all(
        k in data
        for k in [
            queried_acc_metric,
            "scaled_queried_train_time",
            "runtime",
        ]
    ):
        return None, None, None

    acc = np.array(data[queried_acc_metric])
    eval_time = np.array(data["scaled_querried_train_time"]) if "scaled_querried_train_time" in data else np.array(data["scaled_queried_train_time"])
    runtime = np.array(data["runtime"])
    loss = np.array(data.get("train_loss", []))

    # Ensure data arrays are not empty and have compatible lengths
    if not (len(acc) > 0 and len(acc) == len(eval_time) and len(acc) == len(runtime)):
        return None, None, None

    # Handle different method types (one-stage, two-stage, random search)
    is_two_stage = -1 in loss
    is_random_search = len(loss) > 0 and np.all(loss == -1)

    if is_random_search:
        total_time = np.cumsum(eval_time)
    elif is_two_stage:
        stage2_indices = np.where(loss != -1)[0]
        if len(stage2_indices) == 0:
            return None, None, None

        first_stage2_idx = stage2_indices[0]
        stage1_search_cost = runtime[:first_stage2_idx].sum()

        stage2_runtime = runtime[stage2_indices]
        acc = acc[stage2_indices]
        eval_time = eval_time[stage2_indices]

        stage2_cumulative_search_cost = np.cumsum(stage2_runtime)
        total_time = stage1_search_cost + stage2_cumulative_search_cost + eval_time
    else:
        cumulative_search_cost = np.cumsum(runtime)
        total_time = cumulative_search_cost + eval_time

    if len(acc) < 1:
        return None, None, None

    # Ensure random guess uses same units as acc (percent vs fraction)
    num_classes = DATASET_CLASSES.get(dataset, 10)
    acc_is_percent = np.nanmax(acc) > 1.5
    random_guess_acc = (100.0 / num_classes) if acc_is_percent else (1.0 / num_classes)

    # Prepend random guess at time=0 (keeps original sequence order)
    total_time = np.insert(total_time, 0, 0)
    acc = np.insert(acc, 0, random_guess_acc)

    # --- Compute incumbent along the ORIGINAL order (no sorting) ---
    acc_inc = np.maximum.accumulate(acc)

    # --- Collapse duplicate timestamps while preserving the first-occurrence order.
    # For duplicates, keep the maximum incumbent observed for that timestamp.
    seen = {}
    order = []
    for t, a in zip(total_time, acc_inc):
        if t not in seen:
            seen[t] = a
            order.append(t)
        else:
            # update stored value to the maximum incumbent at this timestamp
            if a > seen[t]:
                seen[t] = a

    times_ordered = np.array([float(t) for t in order], dtype=float)
    acc_ordered = np.array([seen[t] for t in order], dtype=float)

    # --- Ensure non-decreasing timestamps for interpolation (preserve order, add tiny eps where needed) ---
    if len(times_ordered) > 1:
        eps = 1e-6  # tiny increment in seconds to enforce strict increase while preserving order
        for i in range(1, len(times_ordered)):
            if times_ordered[i] <= times_ordered[i - 1]:
                times_ordered[i] = times_ordered[i - 1] + eps

    # Calculate run AUC on the chronological incumbent sequence (no re-ordering)
    try:
        run_auc = float(np.trapz(acc_ordered, times_ordered))
    except Exception:
        run_auc = 0.0

    return times_ordered, acc_ordered, run_auc


def plot_anytime_performance(
    root_dir,
    acc_metric="valid_acc",
    show_auc_fill=False,
    show_auc_text=False,
    output_dir="plots",
    combine_plots=True,
):
    """
    Generates and saves anytime performance plots for all optimizers found in root_dir.
    """
    print(f"Searching for runs in: {root_dir}")
    files = find_error_files(root_dir)
    if not files:
        print("No 'errors.json' files found. Exiting.")
        return

    # Group runs by optimizer, zcp_method (if any), dataset, and search space
    grouped_runs = defaultdict(list)
    for f in files:
        key = (f["optimizer"], f.get("zcp_method"), f["dataset"], f["search_space"])
        grouped_runs[key].append(f)

    print(
        f"Found {len(files)} total runs, grouped into {len(grouped_runs)} experiments."
    )

    os.makedirs(output_dir, exist_ok=True)

    # Create a consistent mapping from seed value to marker
    all_seeds = sorted(list(set(f["seed"] for f in files)))
    seed_to_marker = {
        seed: MARKERS[i % len(MARKERS)] for i, seed in enumerate(all_seeds)
    }

    # Consistent color/line per (optimizer, zcp_method) across datasets
    method_keys = sorted(set((opt, zcp) for (opt, zcp, _, _) in grouped_runs.keys()))
    opt_to_color = {
        (opt, zcp): COLORS[i % len(COLORS)] for i, (opt, zcp) in enumerate(method_keys)
    }
    opt_to_fmt = {
        (opt, zcp): FMTS[i % len(FMTS)] for i, (opt, zcp) in enumerate(method_keys)
    }

    if not combine_plots:
        ax = None  # Will be created inside the loop
        # Generate a plot for each group
        for group_idx, (
            (optimizer, zcp_method, dataset, search_space),
            runs,
        ) in enumerate(grouped_runs.items()):
            plt.figure(figsize=(12, 7))
            ax = plt.gca()

            all_aucs = []
            all_trajectories = []
            all_valid_runs_meta = []  # Store metadata for valid runs
            group_label = format_method_label(optimizer, zcp_method)
            color = opt_to_color[(optimizer, zcp_method)]
            fmt = opt_to_fmt[(optimizer, zcp_method)]

            print(f"\nProcessing {group_label} on {dataset} ({search_space})...")

            # Process runs and collect valid data first
            for run_meta in runs:
                time, acc, run_auc = process_run_data(
                    run_meta["path"], acc_metric, dataset
                )
                if time is None:
                    print(f"  - Skipping seed {run_meta['seed']} (no data).")
                    continue

                all_aucs.append(run_auc)
                all_trajectories.append((time, acc))
                all_valid_runs_meta.append(run_meta)

            if not all_trajectories:
                print("  - No valid data for this group. Skipping plot.")
                if not combine_plots:
                    plt.close()
                continue

            # Generate shades for individual seed runs
            num_seeds = len(all_trajectories)
            seed_colors = get_color_shades(color, num_seeds)

            # Create a common time grid for interpolation for all lines (mean and seeds)
            max_time = (
                max(t[-1] for t, a in all_trajectories) if all_trajectories else 0
            )
            time_grid = np.linspace(0, max_time, 500)

            # Plot individual runs from the collected valid data
            if not combine_plots:
                # For individual plots, label each seed clearly
                for i, (time, acc) in enumerate(all_trajectories):
                    run_meta = all_valid_runs_meta[i]
                    marker = seed_to_marker.get(run_meta["seed"], "x")

                    # Interpolate seed data for a smooth curve
                    unique_indices = np.unique(time, return_index=True)[1]
                    interp_acc = np.interp(
                        time_grid, time, acc
                    )

                    # Adjust marker size and width based on the marker type
                    current_markersize = 6 if marker == "+" else 3
                    current_markeredgewidth = 1 if marker == "+" else 1

                    # --- Split plot for interpolated vs. real data ---
                    first_real_time = time[1]
                    split_idx = np.searchsorted(time_grid, first_real_time)

                    # Plot the "estimated" part (lighter)
                    ax.plot(
                        time_grid[: split_idx + 1],
                        interp_acc[: split_idx + 1],
                        color=seed_colors[i],
                        alpha=0.2,  # Lighter
                        linestyle=":",
                        label=None,  # No legend entry for the line itself
                    )
                    # Plot the "real" part (heavier)
                    ax.plot(
                        time_grid[split_idx:],
                        interp_acc[split_idx:],
                        color=seed_colors[i],
                        alpha=0.9,  # Heavier
                        linestyle=":",
                        label=None,  # No extra legend entry
                    )

                    # Overlay the original data points as markers
                    # Draw compact markers with thin white edge + tiny jitter to reduce exact overlap
                    plot_scatter_markers(
                        ax,
                        time[1:],  # Exclude the t=0 random guess point
                        acc[1:],
                        color=seed_colors[i],
                        marker=marker,
                        seed=run_meta.get("seed", i),
                        max_time=max_time,
                        jitter_frac=1e-4,
                        size=24,
                        alpha=0.85,
                        edgecolor="white",
                        linewidth=0.4,
                        rasterize=True,
                    )
            else:
                # For combined plot, use method-specific colors for seeds, but a single legend entry
                for i, (time, acc) in enumerate(all_trajectories):
                    run_meta = all_valid_runs_meta[i]
                    marker = seed_to_marker.get(run_meta["seed"], "x")

                    # Interpolate seed data for a smooth curve
                    unique_indices = np.unique(time, return_index=True)[1]
                    interp_acc = np.interp(
                        time_grid, time, acc
                    )

                    # Adjust marker size and width for '+'
                    current_markersize = 6 if marker == "+" else 3
                    current_markeredgewidth = 1 if marker == "+" else 1

                    # --- Split plot for interpolated vs. real data ---
                    first_real_time = time[1]
                    split_idx = np.searchsorted(time_grid, first_real_time)

                    # Plot the "estimated" part (lighter)
                    ax.plot(
                        time_grid[: split_idx + 1],
                        interp_acc[: split_idx + 1],
                        color=seed_colors[i],
                        alpha=0.2,  # Lighter
                        linestyle=":",
                        linewidth=1.2,
                        label=None,
                    )
                    # Plot the "real" part (heavier)
                    ax.plot(
                        time_grid[split_idx:],
                        interp_acc[split_idx:],
                        color=seed_colors[i],
                        alpha=0.9,  # Heavier
                        linestyle=":",
                        linewidth=1.2,
                        label=None,
                    )

                    # Overlay the original data points as markers
                    # Draw compact markers with thin white edge + tiny jitter to reduce exact overlap
                    plot_scatter_markers(
                        ax,
                        time[1:],  # Exclude the t=0 random guess point
                        acc[1:],
                        color=seed_colors[i],
                        marker=marker,
                        seed=run_meta.get("seed", i),
                        max_time=max_time,
                        jitter_frac=1e-4,
                        size=24,
                        alpha=0.85,
                        edgecolor="white",
                        linewidth=0.4,
                        rasterize=True,
                    )

            # --- Aggregation and Mean Plot ---
            # Create a common time grid for interpolation
            if (
                not all_trajectories
            ):  # Recalculate max_time if needed, though it should exist
                max_time = 0
            else:
                max_time = max(t[-1] for t, a in all_trajectories)
            time_grid = np.linspace(0, max_time, 500)

            interpolated_accs = []
            for time, acc in all_trajectories:
                # Ensure time is monotonically increasing for interpolation
                unique_indices = np.unique(time, return_index=True)[1]
                interp_acc = np.interp(
                    time_grid, time[unique_indices], acc[unique_indices]
                )
                interpolated_accs.append(interp_acc)

            mean_acc = np.mean(interpolated_accs, axis=0)
            std_acc = np.std(interpolated_accs, axis=0)

            # --- AUC Reporting ---
            mean_auc = np.mean(all_aucs)
            std_auc = np.std(all_aucs)
            print(f"  - AUC: {mean_auc:.2f} ± {std_auc:.2f}")
            for i, run_auc in enumerate(all_aucs):
                print(f"    - Seed {all_valid_runs_meta[i]['seed']}: {run_auc:.2f}")

            # --- Plotting Mean and Std Dev ---
            # Construct the label for the legend
            num_seeds = len(all_trajectories)
            mean_label = f"{group_label}"
            if show_auc_text:
                mean_label += f" | AUC: {mean_auc:.2f} ± {std_auc:.2f}"

            # --- Split mean plot for interpolated vs. real data ---
            # Find the average time of the first real data point to split the mean plot
            first_real_times = [t[1] for t, a in all_trajectories if len(t) > 1]
            if first_real_times:
                avg_first_real_time = np.mean(first_real_times)
                mean_split_idx = np.searchsorted(time_grid, avg_first_real_time)
            else:
                mean_split_idx = 0  # Default to no split if no data

            # Plot the "estimated" part of the mean (lighter)
            ax.plot(
                time_grid[: mean_split_idx + 1],
                mean_acc[: mean_split_idx + 1],
                color=color,
                linestyle=fmt,
                linewidth=2.5,
                alpha=0.5,  # Lighter mean line
                label=None,  # Label moved to the 'real' part
            )
            # Plot the "real" part of the mean (heavier)
            ax.plot(
                time_grid[mean_split_idx:],
                mean_acc[mean_split_idx:],
                color=color,
                linestyle=fmt,
                linewidth=2.5,
                alpha=1.0,  # Heavier mean line
                label=mean_label,
            )

            # Split the std. dev. fill to match the mean line's alpha
            ax.fill_between(
                time_grid[: mean_split_idx + 1],
                (mean_acc - std_acc)[: mean_split_idx + 1],
                (mean_acc + std_acc)[: mean_split_idx + 1],
                color=color,
                alpha=0.1,  # Lighter fill
                label=None,
            )
            ax.fill_between(
                time_grid[mean_split_idx:],
                (mean_acc - std_acc)[mean_split_idx:],
                (mean_acc + std_acc)[mean_split_idx:],
                color=color,
                alpha=0.2,  # Heavier fill
                label=None,
            )

            if show_auc_fill:
                ax.fill_between(
                    time_grid, 0, mean_acc, color=color, alpha=0.1, label=None
                )

            # --- Final Plot Configuration (for individual plots) ---
            if not combine_plots:
                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch

                # Get existing handles and labels (should just be the mean line)
                handles, labels = ax.get_legend_handles_labels()

                # Create a handle for the standard deviation fill
                std_dev_handle = Patch(facecolor=color, alpha=0.2, label="Std. Dev.")

                # Create custom legend handles for each seed's marker
                seed_handles = []
                seed_labels = []
                # Sort by seed number for consistent legend order
                for run_meta in sorted(
                    all_valid_runs_meta, key=lambda x: int(x["seed"])
                ):
                    seed = run_meta["seed"]
                    marker = seed_to_marker.get(seed, "x")
                    # Create a handle for each seed with the method's color
                    seed_handles.append(
                        Line2D(
                            [0],
                            [0],
                            linestyle="None",
                            marker=marker,
                            color=color,
                            markersize=8,
                        )
                    )
                    seed_labels.append(f"{seed}")

                # Combine handles and create the legend in the desired order
                ax.legend(
                    handles=handles + [std_dev_handle] + seed_handles,
                    labels=labels + ["Std. Dev."] + seed_labels,
                    loc="lower right",
                    ncol=1,  # vertical
                )

                ax.set_xlabel("Runtime (s) [Linear Scale]")
                if acc_metric == "valid_acc":
                    ax.set_title(
                        f"Incumbent Anytime Validation Performance | {dataset.upper()} | NAS-Bench-201"
                    )
                    ax.set_ylabel("Incumbent Validation Accuracy (%) [Linear Scale]")
                else:
                    ax.set_title(
                        f"Incumbent Anytime Training Performance | {dataset.upper()} | NAS-Bench-201"
                    )
                    ax.set_ylabel("Incumbent Training Accuracy (%) [Linear Scale]")
                ax.set_xscale("linear")
                ax.set_xlim(left=0)  # Start x-axis at 0 for linear scale
                ax.set_ylim(bottom=0)  # Start y-axis at 0
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                ax.grid(True, which="both", ls="-", alpha=0.5)

                filename = (
                    f"performance_{optimizer}_{dataset}_{search_space}_{acc_metric}"
                    + (f"_{zcp_method}" if zcp_method else "")
                    + ".png"
                )
                save_path = os.path.join(output_dir, filename)
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()
                print(f"  - Plot saved to {save_path}")
        return

    # --- Combined mode: one figure per dataset ---
    datasets = sorted(set(f["dataset"] for f in files))
    for dataset in datasets:
        print(f"\nCreating combined plot for dataset: {dataset}")
        fig = plt.figure(figsize=(14, 8))
        ax = plt.gca()

        # Collect groups of this dataset only
        ds_groups = [
            ((optimizer, zcp_method, ds, search_space), runs)
            for (
                (optimizer, zcp_method, ds, search_space),
                runs,
            ) in grouped_runs.items()
            if ds == dataset
        ]

        # --- Pre-scan to find the global max time for this dataset for AUC normalization ---
        global_max_time = 0
        all_run_data = {}  # Cache processed data to avoid re-reading files
        for (optimizer, zcp_method, ds, search_space), runs in ds_groups:
            group_key = (optimizer, zcp_method, ds, search_space)
            all_run_data[group_key] = []
            for run_meta in runs:
                time, acc, _ = process_run_data(run_meta["path"], acc_metric, dataset)
                if time is not None and len(time) > 0:
                    global_max_time = max(global_max_time, time[-1])
                    all_run_data[group_key].append(
                        {"meta": run_meta, "time": time, "acc": acc}
                    )
        print(
            f"  Global max time for {dataset} set to {global_max_time:.2f}s for AUC normalization."
        )

        # Keep track for legend handles later
        all_handles = []
        all_labels = []

        # Process each optimizer group for this dataset
        for (optimizer, zcp_method, ds, search_space), _ in ds_groups:
            group_key = (optimizer, zcp_method, ds, search_space)
            processed_runs = all_run_data[group_key]

            all_norm_aucs = []
            all_trajectories = []
            all_valid_runs_meta = []
            group_label = format_method_label(optimizer, zcp_method)
            color = opt_to_color[(optimizer, zcp_method)]
            fmt = opt_to_fmt[(optimizer, zcp_method)]

            print(f"  Processing {group_label} on {dataset} ({search_space})...")

            # Process runs and collect valid data first
            for run_data in processed_runs:
                time, acc = run_data["time"], run_data["acc"]
                run_meta = run_data["meta"]

                # Calculate NORMALIZED AUC
                run_norm_auc = calculate_normalized_auc(time, acc, global_max_time)

                all_norm_aucs.append(run_norm_auc)
                all_trajectories.append((time, acc))
                all_valid_runs_meta.append(run_meta)

            if not all_trajectories:
                print("    - No valid data for this group. Skipping.")
                continue

            # Generate shades for individual seed runs
            num_seeds = len(all_trajectories)
            seed_colors = get_color_shades(color, num_seeds)

            # Common time grid
            max_time = max(t[-1] for t, a in all_trajectories)
            time_grid = np.linspace(0, max_time, 500)

            # Plot individual runs (combined style)
            for i, (time, acc) in enumerate(all_trajectories):
                run_meta = all_valid_runs_meta[i]
                marker = seed_to_marker.get(run_meta["seed"], "x")

                # Interpolate seed data for a smooth curve
                unique_indices = np.unique(time, return_index=True)[1]
                interp_acc = np.interp(
                    time_grid, time, acc
                )

                # Marker styling
                current_markersize = 6 if marker == "+" else 3
                current_markeredgewidth = 1 if marker == "+" else 1

                first_real_time = time[1]
                split_idx = np.searchsorted(time_grid, first_real_time)

                # Estimated part
                ax.plot(
                    time_grid[: split_idx + 1],
                    interp_acc[: split_idx + 1],
                    color=seed_colors[i],
                    alpha=0.2,
                    linestyle=":",
                    linewidth=1.2,
                    label=None,
                )
                # Real part
                ax.plot(
                    time_grid[split_idx:],
                    interp_acc[split_idx:],
                    color=seed_colors[i],
                    alpha=0.9,
                    linestyle=":",
                    linewidth=1.2,
                    label=None,
                )
                # Markers
                # Draw compact markers with thin white edge + tiny jitter to reduce exact overlap
                plot_scatter_markers(
                    ax,
                    time[1:],  # Exclude the t=0 random guess point
                    acc[1:],
                    color=seed_colors[i],
                    marker=marker,
                    seed=run_meta.get("seed", i),
                    max_time=max_time,
                    jitter_frac=1e-4,
                    size=20,
                    alpha=0.75,
                    edgecolor="white",
                    linewidth=0.35,
                    rasterize=True,
                )

            # Mean and std on common grid
            interpolated_accs = []
            for time, acc in all_trajectories:
                # process_run_data returns time-ordered, deduplicated arrays (preserving original order),
                # so interpolate directly onto the common grid
                interp_acc = np.interp(time_grid, time, acc)
                interpolated_accs.append(interp_acc)

            mean_acc = np.mean(interpolated_accs, axis=0)
            std_acc = np.std(interpolated_accs, axis=0)

            mean_auc = np.mean(all_norm_aucs)
            std_auc = np.std(all_norm_aucs)
            print(f"    - Normalized AUC: {mean_auc:.4f} ± {std_auc:.4f}")
            for i, run_auc in enumerate(all_norm_aucs):
                print(f"      - Seed {all_valid_runs_meta[i]['seed']}: {run_auc:.4f}")

            first_real_times = [t[1] for t, a in all_trajectories if len(t) > 1]
            mean_split_idx = (
                np.searchsorted(time_grid, np.mean(first_real_times))
                if first_real_times
                else 0
            )

            mean_label = f"{group_label}"
            if show_auc_text:
                mean_label += f" | AUC: {mean_auc:.2f} ± {std_auc:.2f}"

            # Mean line (estimated + real)
            ax.plot(
                time_grid[: mean_split_idx + 1],
                mean_acc[: mean_split_idx + 1],
                color=color,
                linestyle=fmt,
                linewidth=2.5,
                alpha=0.5,
                label=None,
            )
            handle = ax.plot(
                time_grid[mean_split_idx:],
                mean_acc[mean_split_idx:],
                color=color,
                linestyle=fmt,
                linewidth=2.5,
                alpha=1.0,
                label=mean_label,
            )[0]

            # Std fill (estimated + real)
            ax.fill_between(
                time_grid[: mean_split_idx + 1],
                (mean_acc - std_acc)[: mean_split_idx + 1],
                (mean_acc + std_acc)[: mean_split_idx + 1],
                color=color,
                alpha=0.1,
                label=None,
            )
            ax.fill_between(
                time_grid[mean_split_idx:],
                (mean_acc - std_acc)[mean_split_idx:],
                (mean_acc + std_acc)[mean_split_idx:],
                color=color,
                alpha=0.2,
                label=None,
            )

            if show_auc_fill:
                ax.fill_between(time_grid, 0, mean_acc, color=color, alpha=0.1)

        # Build legend (method lines + std + seed markers)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles, labels = ax.get_legend_handles_labels()
        std_dev_handle = Patch(facecolor="gray", alpha=0.4, label="Std. Dev.")

        seed_handles = []
        seed_labels = []
        for seed, marker in sorted(seed_to_marker.items()):
            seed_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=marker,
                    color="gray",
                    markersize=8,
                )
            )
            seed_labels.append(f"{seed}")

        ax.legend(
            handles=handles + [std_dev_handle] + seed_handles,
            labels=labels + ["Std. Dev."] + seed_labels,
            loc="lower right",
            ncol=1,
        )

        ax.set_xlabel("Runtime (s) [Linear Scale]")
        if acc_metric == "valid_acc":
            ax.set_ylabel("Incumbent Validation Accuracy (%) [Linear Scale]")
            plot_title = f"Incumbent Anytime Validation Performance | {dataset.upper()} | NAS-Bench-201"
        else:
            ax.set_ylabel("Incumbent Training Accuracy (%) [Linear Scale]")
            plot_title = f"Incumbent Anytime Training Performance | {dataset.upper()} | NAS-Bench-201"
        ax.set_title(plot_title)
        ax.set_xscale("linear")
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, which="both", ls="-", alpha=0.5)

        filename = f"combined_performance_plot_{acc_metric}_{dataset}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Combined plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate anytime performance plots from NASLib experiment directories."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp",
        help="The root directory containing the experiment runs.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="valid_acc",
        choices=["train_acc", "valid_acc"],
        help="The accuracy metric to plot.",
    )
    parser.add_argument(
        "--show_auc_fill",
        action="store_true",
        help="If set, shades the area under the mean performance curve.",
    )
    parser.add_argument(
        "--show_auc_text",
        action="store_true",
        help="If set, displays the Mean AUC value on the plot.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/plotting_scripts/plots",
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--combine_plots",
        action="store_true",
        help="If set, combines all optimizer results into a single plot.",
    )
    # python naslib/optimizers/oneshot/gsparsity/final_hp_setting_performance_plotting.py --combine_plots --show_auc_text --out_dir naslib/optimizers/oneshot/gsparsity/final_hp_visualization
    parser.set_defaults(combine_plots=False, show_auc_text=False, show_auc_fill=False)
    args = parser.parse_args()

    plot_anytime_performance(
        root_dir=args.root_dir,
        acc_metric=args.metric,
        show_auc_fill=args.show_auc_fill,
        show_auc_text=args.show_auc_text,
        output_dir=args.out_dir,
        combine_plots=args.combine_plots,
    )


if __name__ == "__main__":
    main()

