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
from sklearn.metrics import auc
import argparse
import re
from collections import defaultdict
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# Dataset classes for random guess calculation
DATASET_CLASSES = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}

# Default color and style settings for plots
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
# New color palette provided by the user, converted from [0, 255] to [0, 1] range
DEFAULTS = [
    (0.9411764705882353, 0.27450980392156865, 0.7490196078431373),  # [240,70,191]
    (0.3843137254901961, 0.7686274509803922, 0.20784313725490197),  # [98,196,53]
    (0.35294117647058826, 0.25098039215686274, 0.6549019607843137),  # [90,64,167]
    (0.43529411764705883, 0.8627450980392157, 0.5607843137254902),  # [111,220,143]
    (0.6313725490196078, 0.0784313725490196, 0.27450980392156865),  # [161,20,70]
    (0.00784313725490196, 0.807843137254902, 0.9803921568627451),  # [2,206,250]
    (0.9529411764705882, 0.3607843137254902, 0.1568627450980392),  # [243,92,40]
    (0.2196078431372549, 0.3607843137254902, 0.12941176470588237),  # [56,92,33]
    (1.0, 0.592156862745098, 0.4196078431372549),  # [255,151,107]
    (0.7019607843137254, 0.7803921568627451, 0.5647058823529412),  # [179,199,144]
    (0.7098039215686275, 0.5529411764705883, 0.0),  # [181,141,0]
]
C_MAX = len(DEFAULTS)
COLORS = [*DEFAULTS[:C_MAX]] * 3
FMTS = [*["-"] * C_MAX, *["--"] * C_MAX, *[":"] * C_MAX]
# Prioritize the most visually distinct markers for the first few seeds.
# Circle, Square, Plus, Diamond, and Cross are highly discriminable.
MARKERS = ["o", "s", "+", "D", "x", "^", "*", "v", "<", ">", "p", "h", "H", "P"]

# NEW: methods that receive one-time shift
ZCP_PRE_METHODS = {
    "zcp-pre_gsparsity",
    "zcp-pre_zcp_gsparsity",
}


def load_zcp_pre_durations(durations_dir):
    """
    Load per-dataset zc-scoring durations to be used as a one-time offset.
    Files:
      - arch_scores_duration_cifar10.json
      - arch_scores_duration_cifar100.json
      - arch_scores_duration_ImageNet16-120.json
    Content: {"duration": 13760.6559}
    """
    ds_names = ["cifar10", "cifar100", "ImageNet16-120"]
    mapping = {}
    for ds in ds_names:
        fname = f"arch_scores_duration_{ds}.json"
        fpath = os.path.join(durations_dir, fname)
        try:
            if os.path.isfile(fpath):
                with open(fpath, "r") as f:
                    obj = json.load(f)
                mapping[ds] = float(obj.get("duration", 0.0))
            else:
                mapping[ds] = 0.0
        except Exception:
            mapping[ds] = 0.0
    return mapping


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
    Compact markers for dense stability plots:
      - deterministic jitter (per-seed) to break exact overlap
      - thin contrasting edge (edgecolor) so small markers remain visible
      - rasterized scatter for dense regions to keep vector output small
    jitter_frac: fraction of max_time used as jitter amplitude
    """
    if len(x) == 0:
        return
    try:
        seed_int = (
            int(seed)
            if isinstance(seed, (int, str)) and str(seed).isdigit()
            else hash(seed) & 0xFFFFFFFF
        )
    except Exception:
        seed_int = 0
    prng = np.random.RandomState(seed_int)
    jitter_amount = jitter_frac * (max_time if max_time > 0 else 1.0)
    jitter = (prng.rand(len(x)) - 0.5) * jitter_amount
    x_j = np.array(x) + jitter

    # Make '+' markers more visible: larger size, thicker stroke and high-contrast edge
    if marker == "+":
        size = max(size, 45)  # substantially larger
        linewidth = max(linewidth, 0.5)
        edgecolor = "black"
        alpha = max(alpha, 0.95)
    else:
        # keep a modest white edge for other markers so small markers remain readable
        edgecolor = edgecolor
        linewidth = linewidth

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
    """Finds all 'errors.json' files and extracts metadata from their paths.

    Expected relative structures:
      - With ZCP method: {optimizer}/{zcp_method}/{search_space}/{dataset}/{seed}/errors.json
      - Without ZCP:     {optimizer}/{search_space}/{dataset}/{seed}/errors.json
    """
    error_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "errors.json" in filenames:
            full_path = os.path.join(dirpath, "errors.json")
            relative_path = os.path.relpath(full_path, root_dir)
            parts = relative_path.split(os.sep)

            # Ensure at least optimizer/search_space/dataset/seed/errors.json
            if len(parts) >= 5:
                optimizer = parts[0]
                seed = parts[-2]
                dataset = parts[-3]
                search_space = parts[-4]
                # Detect zcp_method when there is an extra component after optimizer
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


def process_run_data(filepath, acc_metric, dataset, zcp_pre_offset=0.0):
    """
    Loads and processes a single run from an errors.json file.
    Use the queried values (same keys as the performance script) for non-random-search runs,
    but for random search use the raw 'valid_acc' (or 'train_acc' if requested).
    Optionally shifts all real datapoints by a one-time offset (zcp-pre).
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    # Pull runtime and loss first so we can detect random search and two-stage runs
    runtime = np.array(data.get("runtime", []))
    loss = np.array(data.get("train_loss", []))

    # Decide which accuracy series to use
    # - For random search use raw 'valid_acc'/'train_acc' if present
    # - Otherwise use 'queried_val_acc'/'queried_train_acc'
    is_random_search = len(loss) > 0 and np.all(loss == -1)

    if acc_metric == "valid_acc":
        acc_key = (
            "valid_acc"
            if is_random_search and "valid_acc" in data
            else "queried_val_acc"
        )
    else:  # acc_metric == "train_acc"
        acc_key = (
            "train_acc"
            if is_random_search and "train_acc" in data
            else "queried_train_acc"
        )

    acc = np.array(data.get(acc_key, []))

    # Required keys check
    if runtime.size == 0 or acc.size == 0:
        return None, None, None

    # handle possible misspelling used elsewhere
    eval_time = (
        np.array(data["scaled_querried_train_time"])
        if "scaled_querried_train_time" in data
        else np.array(data.get("scaled_queried_train_time", []))
    )

    # basic length checks: prefer eval_time when present (some methods may not have it)
    if eval_time.size and not (len(acc) == len(eval_time) == len(runtime)):
        return None, None, None
    if not eval_time.size and not (len(acc) == len(runtime)):
        return None, None, None

    # Handle different method styles (random search / two-stage / normal)
    # note: keep two-stage flag distinct; random search already handled above
    is_two_stage = -1 in loss

    if is_random_search:
        total_time = np.cumsum(eval_time) if eval_time.size else np.cumsum(runtime)
    elif is_two_stage:
        stage2_indices = np.where(loss != -1)[0]
        if len(stage2_indices) == 0:
            return None, None, None
        first_stage2_idx = stage2_indices[0]
        stage1_search_cost = runtime[:first_stage2_idx].sum()
        stage2_runtime = runtime[stage2_indices]
        acc = acc[stage2_indices]
        eval_time = (
            eval_time[stage2_indices]
            if eval_time.size
            else np.zeros_like(stage2_runtime)
        )
        stage2_cumulative_search_cost = np.cumsum(stage2_runtime)
        total_time = stage1_search_cost + stage2_cumulative_search_cost + eval_time
    else:
        cumulative_search_cost = np.cumsum(runtime)
        total_time = cumulative_search_cost + (eval_time if eval_time.size else 0)

    # NEW: apply one-time shift for zcp-pre methods (only to real datapoints, not the t=0 point)
    if zcp_pre_offset and zcp_pre_offset > 0:
        total_time = total_time + float(zcp_pre_offset)

    if len(acc) < 1:
        return None, None, None

    # Random guess prepend (preserve raw units like queried acc)
    num_classes = DATASET_CLASSES.get(dataset, 10)
    acc_is_percent = np.nanmax(acc) > 1.5
    random_guess_acc = (100.0 / num_classes) if acc_is_percent else (1.0 / num_classes)

    total_time = np.insert(total_time, 0, 0)
    acc = np.insert(acc, 0, random_guess_acc)

    # Collapse duplicate timestamps: preserve first-occurrence order but keep latest raw value
    seen = {}
    order = []
    for t, a in zip(total_time, acc):
        if t not in seen:
            order.append(t)
        seen[t] = a

    times_ordered = np.array([float(t) for t in order], dtype=float)
    acc_ordered = np.array([seen[t] for t in order], dtype=float)

    # enforce strictly increasing times for interpolation
    if len(times_ordered) > 1:
        eps = 1e-6
        for i in range(1, len(times_ordered)):
            if times_ordered[i] <= times_ordered[i - 1]:
                times_ordered[i] = times_ordered[i - 1] + eps

    # Compute run AUC on the selected values
    try:
        run_auc = float(np.trapz(acc_ordered, times_ordered))
    except Exception:
        run_auc = 0.0

    return times_ordered, acc_ordered, run_auc


def plot_anytime_stability(
    root_dir,
    acc_metric="valid_acc",
    show_auc_fill=False,
    show_auc_text=False,
    output_dir="plots",
    combine_plots=True,
    durations_dir="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor",
    xscale="linear",  # NEW
):
    """
    Generates and saves anytime stability plots for all optimizers found in root_dir.
    """
    # NEW: load zcp-pre durations once
    durations_map = load_zcp_pre_durations(durations_dir)
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
        # per-group plotting with zcp support
        for group_idx, (
            (optimizer, zcp_method, dataset, search_space),
            runs,
        ) in enumerate(grouped_runs.items()):
            plt.figure(figsize=(12, 7))
            ax = plt.gca()

            all_aucs = []
            all_trajectories = []
            all_valid_runs_meta = []
            group_label = format_method_label(optimizer, zcp_method)
            color = opt_to_color.get((optimizer, zcp_method), COLORS[0])
            fmt = opt_to_fmt.get((optimizer, zcp_method), FMTS[0])

            print(f"\nProcessing {optimizer} on {dataset} ({search_space})...")

            zcp_offset = (
                float(durations_map.get(str(dataset), 0.0))
                if optimizer in ZCP_PRE_METHODS
                else 0.0
            )

            # Process runs and collect valid data first
            for run_meta in runs:
                time, acc, run_auc = process_run_data(
                    run_meta["path"], acc_metric, dataset, zcp_pre_offset=zcp_offset
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

            # Create a common time grid for interpolation
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

                    # Adjust marker size and width based on the marker type
                    current_markersize = 8 if marker == "+" else 5
                    current_markeredgewidth = 2 if marker == "+" else 1

                    # NEW: estimated segment from t=0 (random guess) to first real datapoint
                    if len(time) > 1:
                        ax.plot(
                            [time[0], time[1]],
                            [acc[0], acc[1]],
                            color=seed_colors[i],
                            linewidth=1.2 if marker == "+" else 1.0,
                            alpha=0.55,
                            linestyle=":",
                            zorder=3,
                        )

                    # Draw a thin line connecting the seed's datapoints (exclude t=0 random guess)
                    if len(time) > 1:
                        ax.plot(
                            time[1:],
                            acc[1:],
                            color=seed_colors[i],
                            linewidth=1.5 if marker == "+" else 1.0,
                            alpha=0.6,
                            linestyle="-",
                            zorder=4,
                        )

                    # Plot the raw data points as markers
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

                    current_markersize = 8 if marker == "+" else 5
                    current_markeredgewidth = 2 if marker == "+" else 1

                    # NEW: estimated segment from t=0 (random guess) to first real datapoint
                    if len(time) > 1:
                        ax.plot(
                            [time[0], time[1]],
                            [acc[0], acc[1]],
                            color=seed_colors[i],
                            linewidth=1.2 if marker == "+" else 1.0,
                            alpha=0.55,
                            linestyle=":",
                            zorder=3,
                        )

                    # Draw a thin line connecting the seed's datapoints (exclude t=0 random guess)
                    if len(time) > 1:
                        ax.plot(
                            time[1:],
                            acc[1:],
                            color=seed_colors[i],
                            linewidth=1.5 if marker == "+" else 1.0,
                            alpha=0.6,
                            linestyle="-",
                            zorder=4,
                        )

            # --- Aggregation and Mean Plot ---
            # Create a common time grid for interpolation
            interpolated_accs = []
            for time, acc in all_trajectories:
                # Ensure time is monotonically increasing for interpolation
                unique_indices = np.unique(time, return_index=True)[1]
                interp_acc = np.interp(
                    time_grid, time[unique_indices], acc[unique_indices]
                )
                interpolated_accs.append(interp_acc)

            # Calculate min and max performance across seeds for the stability area
            min_acc = np.min(interpolated_accs, axis=0)
            max_acc = np.max(interpolated_accs, axis=0)

            # --- Weighted instability AUC (give real data more weight) ---
            # Each seed contributes "real support" only after its first real data point (t[1]).
            first_real_times = [t[1] for t, a in all_trajectories if len(t) > 1]
            if first_real_times:
                earliest_first = float(np.min(first_real_times))
                frt_arr = np.array(first_real_times)
                # weights(t) = fraction of seeds that have produced at least one real data point by time t
                weights = np.mean(time_grid[:, None] >= frt_arr[None, :], axis=1)
            else:
                earliest_first = 0.0
                weights = np.ones_like(time_grid)

            # Unweighted (legacy) and weighted instability
            unweighted_instability_auc = auc(time_grid, max_acc) - auc(
                time_grid, min_acc
            )
            weighted_band = (max_acc - min_acc) * weights
            instability_auc = auc(time_grid, weighted_band)

            print(
                f"  - Instability Area (weighted): {instability_auc:.2f} | (unweighted): {unweighted_instability_auc:.2f}"
            )

            # --- Plotting Stability Area with improved split ---
            # Start the "real" band exactly at the earliest real data point across seeds
            split_idx = int(np.searchsorted(time_grid, earliest_first))

            # Build instability legend entry but only draw the fill if requested.
            instability_label = f"{group_label}"
            if show_auc_text:
                instability_label += f" | Instab. AUC (w): {instability_auc:.2f}"

            if show_auc_fill:
                # preview before first real point (very light) and main band
                if split_idx > 0:
                    ax.fill_between(
                        time_grid[: split_idx + 1],
                        min_acc[: split_idx + 1],
                        max_acc[: split_idx + 1],
                        color=color,
                        alpha=0.08,
                        label=None,
                    )
                ax.fill_between(
                    time_grid[split_idx:],
                    min_acc[split_idx:],
                    max_acc[split_idx:],
                    color=color,
                    alpha=0.30,
                    label=instability_label,
                )
                instability_handle = Patch(facecolor=color, alpha=0.3)
            else:
                # Do not draw the colored band on the axes; still provide a legend handle
                # that shows the method color (outline) so the legend indicates the area.
                instability_handle = Patch(
                    facecolor="none", edgecolor=color, linewidth=1.5
                )

            # --- Final Plot Configuration (for individual plots) ---
            if not combine_plots:
                from matplotlib.lines import Line2D

                # Use our prepared instability_handle (band may or may not have been drawn)
                handles = [instability_handle]

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

                # Combine handles and create the legend
                ax.legend(
                    handles=handles + seed_handles,
                    labels=[instability_label] + seed_labels,
                    loc="upper left",
                    ncol=1,
                )

                ax.set_xlabel(
                    f"Normalized Time (s) [{'Log' if xscale == 'log' else 'Linear'} Scale]"
                )
                if acc_metric == "valid_acc":
                    ax.set_ylabel("Raw Validation Accuracy (%) [Linear Scale]")
                    ax.set_title(
                        f"Anytime Raw Validation Stability | {dataset.upper()} | NAS-Bench-201"
                    )
                else:
                    ax.set_ylabel("Raw Training Accuracy (%) [Linear Scale]")
                    ax.set_title(
                        f"Anytime Raw Training Stability | {dataset.upper()} | NAS-Bench-201"
                    )
                ax.set_xscale(xscale)
                if xscale == "log":
                    # Linear region up to the first real datapoint across seeds
                    min_pos_time = np.inf
                    for t, _ in all_trajectories:
                        tp = np.asarray(t)
                        tp = tp[tp > 0]
                        if tp.size:
                            min_pos_time = min(min_pos_time, float(tp.min()))
                    linthresh = (
                        float(min_pos_time) if np.isfinite(min_pos_time) else 1.0
                    )
                    try:
                        ax.set_xscale("symlog", linthresh=linthresh)
                    except TypeError:
                        ax.set_xscale("symlog", linthreshx=linthresh)
                    ax.set_xlim(left=0)
                else:
                    ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                ax.grid(True, which="both", ls="-", alpha=0.5)

                filename = (
                    f"stability_{optimizer}_{dataset}_{search_space}_{acc_metric}"
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
        print(f"\nCreating combined stability plot for dataset: {dataset}")
        fig = plt.figure(figsize=(14, 8))
        ax = plt.gca()

        # keep zcp in the grouping for combined plots as well
        ds_groups = [
            ((optimizer, zcp_method, ds, search_space), runs)
            for (
                (optimizer, zcp_method, ds, search_space),
                runs,
            ) in grouped_runs.items()
            if ds == dataset
        ]

        # collect explicit method handles & labels so we can always show method legend entries
        method_handles = []
        method_labels = []
        # iterate with the correct unpacking including zcp_method
        for (optimizer, zcp_method, ds, search_space), runs in ds_groups:
            all_aucs = []
            all_trajectories = []
            all_valid_runs_meta = []
            group_label = format_method_label(optimizer, zcp_method)
            color = opt_to_color.get((optimizer, zcp_method), COLORS[0])

            zcp_offset = (
                float(durations_map.get(str(ds), 0.0))
                if optimizer in ZCP_PRE_METHODS
                else 0.0
            )
            print(f"  Processing {optimizer} on {dataset} ({search_space})...")

            for run_meta in runs:
                time, acc, run_auc = process_run_data(
                    run_meta["path"], acc_metric, dataset, zcp_pre_offset=zcp_offset
                )
                if time is None:
                    print(f"    - Skipping seed {run_meta['seed']} (no data).")
                    continue
                all_aucs.append(run_auc)
                all_trajectories.append((time, acc))
                all_valid_runs_meta.append(run_meta)

            if not all_trajectories:
                print("    - No valid data for this group. Skipping.")
                continue

            # Seed shades
            num_seeds = len(all_trajectories)
            seed_colors = get_color_shades(color, num_seeds)

            max_time = max(t[-1] for t, a in all_trajectories)
            time_grid = np.linspace(0, max_time, 500)

            # Plot individual seeds
            for i, (time, acc) in enumerate(all_trajectories):
                run_meta = all_valid_runs_meta[i]
                marker = seed_to_marker.get(run_meta["seed"], "x")
                current_markersize = 8 if marker == "+" else 5
                current_markeredgewidth = 2 if marker == "+" else 1

                # NEW: estimated segment from t=0 (random guess) to first real datapoint
                if len(time) > 1:
                    ax.plot(
                        [time[0], time[1]],
                        [acc[0], acc[1]],
                        color=seed_colors[i],
                        linewidth=1.2 if marker == "+" else 1.0,
                        alpha=0.55,
                        linestyle=":",
                        zorder=3,
                    )

                # Draw a thin line connecting the seed's datapoints (exclude t=0 random guess)
                if len(time) > 1:
                    ax.plot(
                        time[1:],
                        acc[1:],
                        color=seed_colors[i],
                        linewidth=1.5 if marker == "+" else 1.0,
                        alpha=0.55,
                        linestyle="-",
                        zorder=4,
                    )

            # Interpolate to compute min/max band
            interpolated_accs = []
            for time, acc in all_trajectories:
                unique_indices = np.unique(time, return_index=True)[1]
                interp_acc = np.interp(
                    time_grid, time[unique_indices], acc[unique_indices]
                )
                interpolated_accs.append(interp_acc)

            min_acc = np.min(interpolated_accs, axis=0)
            max_acc = np.max(interpolated_accs, axis=0)

            # Weighted instability (real points count more)
            first_real_times = [t[1] for t, a in all_trajectories if len(t) > 1]
            if first_real_times:
                earliest_first = float(np.min(first_real_times))
                frt_arr = np.array(first_real_times)
                weights = np.mean(time_grid[:, None] >= frt_arr[None, :], axis=1)
            else:
                earliest_first = 0.0
                weights = np.ones_like(time_grid)

            unweighted_instability_auc = auc(time_grid, max_acc) - auc(
                time_grid, min_acc
            )
            weighted_band = (max_acc - min_acc) * weights
            instability_auc = auc(time_grid, weighted_band)
            print(
                f"    - Instability Area (weighted): {instability_auc:.2f} | (unweighted): {unweighted_instability_auc:.2f}"
            )

            instability_label = f"{group_label}"
            if show_auc_text:
                instability_label += f" | Instab. AUC (w): {instability_auc:.2f}"

            # Split fill at earliest real time across seeds (visual fix)
            split_idx = int(np.searchsorted(time_grid, earliest_first))

            # Only draw the band if requested. Always create a legend handle.
            if show_auc_fill:
                if split_idx > 0:
                    ax.fill_between(
                        time_grid[: split_idx + 1],
                        min_acc[: split_idx + 1],
                        max_acc[: split_idx + 1],
                        color=color,
                        alpha=0.08,
                        label=None,
                    )
                ax.fill_between(
                    time_grid[split_idx:],
                    min_acc[split_idx:],
                    max_acc[split_idx:],
                    color=color,
                    alpha=0.30,
                    label=instability_label,
                )
                instability_handle = Patch(facecolor=color, alpha=0.3)
            else:
                instability_handle = Patch(
                    facecolor="none", edgecolor=color, linewidth=1.5
                )

            # collect method handle/label for the combined legend
            method_handles.append(instability_handle)
            method_labels.append(instability_label)
            # Track smallest positive time for later axis limit
            # Add this near where all_trajectories is available
            if "dataset_min_pos_time" not in locals():
                dataset_min_pos_time = np.inf
            for t, _ in all_trajectories:
                tp = np.asarray(t)
                tp = tp[tp > 0]
                if tp.size:
                    dataset_min_pos_time = min(dataset_min_pos_time, float(tp.min()))

        # Legend: methods + seed markers (use collected method_handles so legend entries exist even
        # when bands were not drawn)
        from matplotlib.lines import Line2D

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
            handles=method_handles + seed_handles,
            labels=method_labels + seed_labels,
            loc="upper left",
            ncol=1,
        )

        ax.set_xlabel(
            f"Normalized Time (s) [{'Log' if xscale == 'log' else 'Linear'} Scale]"
        )
        if acc_metric == "valid_acc":
            ax.set_ylabel("Raw Validation Accuracy (%) [Linear Scale]")
            plot_title = (
                f"Anytime Raw Validation Stability | {dataset.upper()} | NAS-Bench-201"
            )
        else:
            ax.set_ylabel("Raw Training Accuracy (%) [Linear Scale]")
            plot_title = (
                f"Anytime Raw Training Stability | {dataset.upper()} | NAS-Bench-201"
            )
        ax.set_title(plot_title)
        ax.set_xscale(xscale)
        if xscale == "log":
            linthresh = (
                float(dataset_min_pos_time)
                if "dataset_min_pos_time" in locals()
                and np.isfinite(dataset_min_pos_time)
                else 1.0
            )
            try:
                ax.set_xscale("symlog", linthresh=linthresh)
            except TypeError:
                ax.set_xscale("symlog", linthreshx=linthresh)
            ax.set_xlim(left=0)
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, which="both", ls="-", alpha=0.5)

        filename = f"combined_stability_plot_{acc_metric}_{dataset}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Combined stability plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate anytime stability plots from NASLib experiment directories."
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
        help="This flag is kept for compatibility but the min-max area is always shown.",
    )
    parser.add_argument(
        "--show_auc_text",
        action="store_true",
        help="If set, displays the Instability Area AUC value in the legend.",
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
    parser.add_argument(
        "--durations_dir",
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor",
        help="Directory containing arch_scores_duration_*.json files for zcp-pre shift.",
    )
    parser.add_argument(
        "--xscale",
        choices=["linear", "log"],
        default="linear",
        help="Scale for the x-axis (time).",
    )
    parser.set_defaults(combine_plots=False, show_auc_text=False, show_auc_fill=False)
    args = parser.parse_args()

    plot_anytime_stability(
        root_dir=args.root_dir,
        acc_metric=args.metric,
        show_auc_fill=args.show_auc_fill,
        show_auc_text=args.show_auc_text,
        output_dir=args.out_dir,
        combine_plots=args.combine_plots,
        durations_dir=args.durations_dir,
        xscale=args.xscale,  # NEW
    )


if __name__ == "__main__":
    main()
