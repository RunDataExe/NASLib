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


def find_error_files(root_dir):
    """Finds all 'errors.json' files and extracts metadata from their paths."""
    error_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "errors.json" in filenames:
            full_path = os.path.join(dirpath, "errors.json")
            # Normalize path for consistent parsing
            relative_path = os.path.relpath(full_path, root_dir)
            parts = relative_path.split(os.sep)

            # Path structure: {optimizer}/{...}/{search_space}/{dataset}/{seed}/errors.json
            if len(parts) >= 5:
                optimizer = parts[0]
                seed = parts[-2]
                dataset = parts[-3]
                search_space = parts[-4]
                # Handle potential extra directories like zcp_method
                zcp_method = parts[1] if "zcp" in optimizer and len(parts) > 5 else None

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
    """Loads and processes a single run from an errors.json file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if not data.get(acc_metric) or not data.get("runtime"):
        return None, None, None

    acc = np.array(data[acc_metric])
    runtime = np.array(data["runtime"])
    loss = np.array(data.get("train_loss", []))  # Use train_loss to detect stages

    # Handle two-stage methods by checking for -1 in loss values
    is_two_stage = -1 in loss
    if is_two_stage:
        stage2_indices = np.where(loss != -1)[0]

        # If there are no entries for stage 2, check if it's a fully queried method
        if len(stage2_indices) == 0:
            # This is a fully queried method if all loss values are -1
            if len(loss) > 0 and np.all(loss == -1):
                # For fully queried methods, time is cumulative runtime
                cumulative_time = np.cumsum(runtime)
            else:
                # It's a two-stage method that didn't reach stage 2, so skip.
                return None, None, None
        else:
            # This is a standard two-stage method
            first_stage2_idx = stage2_indices[0]
            stage1_runtime = runtime[:first_stage2_idx].sum()

            # Filter for stage 2 data
            acc = acc[stage2_indices]
            runtime = runtime[stage2_indices]

            # Prepend the summed stage 1 runtime to the cumulative time of stage 2
            cumulative_time = np.cumsum(runtime) + stage1_runtime
    else:
        # For single-stage methods, just calculate cumulative time
        cumulative_time = np.cumsum(runtime)

    # If after processing, we have less than 1 point, we can't plot or calculate AUC.
    if len(acc) < 1:
        return None, None, None

    # Prepend a random guess at time=0
    num_classes = DATASET_CLASSES.get(dataset, 10)  # Default to 10 if unknown
    random_guess_acc = 1.0 / num_classes
    cumulative_time = np.insert(cumulative_time, 0, 0)
    acc = np.insert(acc, 0, random_guess_acc)

    # Use incumbent (best-so-far) performance
    acc = np.maximum.accumulate(acc)

    # Calculate AUC on the incumbent accuracy, handle cases with a single point.
    run_auc = auc(cumulative_time, acc) if len(cumulative_time) >= 2 else 0.0

    return cumulative_time, acc, run_auc


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

    # Group runs by optimizer, dataset, and search space
    grouped_runs = defaultdict(list)
    for f in files:
        key = (f["optimizer"], f["dataset"], f["search_space"])
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
    if combine_plots:
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        if acc_metric == "valid_acc":
            plot_title = f"Incumbent Anytime Validation Accuracy | {f['dataset'].upper()} | NAS-Bench-201"
        else:
            plot_title = f"Incumbent Anytime Training Accuracy | {f['dataset'].upper()} | NAS-Bench-201"
    else:
        ax = None  # Will be created inside the loop

    # Generate a plot for each group
    for group_idx, ((optimizer, dataset, search_space), runs) in enumerate(
        grouped_runs.items()
    ):
        if not combine_plots:
            plt.figure(figsize=(12, 7))
            ax = plt.gca()

        all_aucs = []
        all_trajectories = []
        all_valid_runs_meta = []  # Store metadata for valid runs
        group_label = f"{optimizer}"
        color = COLORS[group_idx % len(COLORS)]
        fmt = FMTS[group_idx % len(FMTS)]

        print(f"\nProcessing {optimizer} on {dataset} ({search_space})...")

        # Process runs and collect valid data first
        for run_meta in runs:
            time, acc, run_auc = process_run_data(run_meta["path"], acc_metric, dataset)
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
        max_time = max(t[-1] for t, a in all_trajectories) if all_trajectories else 0
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
                    time_grid, time[unique_indices], acc[unique_indices]
                )

                # Adjust marker size and width based on the marker type
                current_markersize = 8 if marker == "+" else 5
                current_markeredgewidth = 2 if marker == "+" else 1  # Make '+' thicker

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
                ax.plot(
                    time[1:],  # Exclude the t=0 random guess point
                    acc[1:],
                    linestyle="None",  # No line connecting markers
                    marker=marker,
                    color=seed_colors[i],
                    markersize=current_markersize,
                    markeredgewidth=current_markeredgewidth,
                    alpha=0.9,
                    label=None,  # No extra legend entry
                )
        else:
            # For combined plot, use method-specific colors for seeds, but a single legend entry
            for i, (time, acc) in enumerate(all_trajectories):
                run_meta = all_valid_runs_meta[i]
                marker = seed_to_marker.get(run_meta["seed"], "x")

                # Interpolate seed data for a smooth curve
                unique_indices = np.unique(time, return_index=True)[1]
                interp_acc = np.interp(
                    time_grid, time[unique_indices], acc[unique_indices]
                )

                # Adjust marker size and width for '+'
                current_markersize = 8 if marker == "+" else 5
                current_markeredgewidth = 2 if marker == "+" else 1

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
                ax.plot(
                    time[1:],  # Exclude the t=0 random guess point
                    acc[1:],
                    linestyle="None",
                    marker=marker,
                    color=seed_colors[i],
                    markersize=current_markersize,
                    markeredgewidth=current_markeredgewidth,
                    alpha=0.7,
                    label=None,
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
            interp_acc = np.interp(time_grid, time[unique_indices], acc[unique_indices])
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
            ax.fill_between(time_grid, 0, mean_acc, color=color, alpha=0.1, label=None)

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
            for run_meta in sorted(all_valid_runs_meta, key=lambda x: int(x["seed"])):
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
                loc="upper left",
                ncol=1,  # vertical
            )

            ax.set_xlabel("Runtime (Seconds) [Log Scale]")
            if acc_metric == "valid_acc":
                ax.set_title(
                    f"Incumbent Anytime Validation Accuracy | {f['dataset'].upper()} | NAS-Bench-201"
                )
                ax.set_ylabel("Validation Accuracy (Percent) [Linear Scale]")
            else:
                ax.set_title(
                    f"Incumbent Anytime Training Accuracy | {f['dataset'].upper()} | NAS-Bench-201"
                )
                ax.set_ylabel("Training Accuracy (Percent) [Linear Scale]")
            ax.set_xscale("log")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.grid(True, which="both", ls="-", alpha=0.5)

            # Add AUC info to plot if requested - This is now handled by the legend
            # if show_auc_text:
            #     auc_text = f"Mean AUC: {mean_auc:.2f} ± {std_auc:.2f}"
            #     plt.figtext(
            #         0.5,
            #         0.01,
            #         auc_text,
            #         ha="center",
            #         fontsize=10,
            #         bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
            #     )

            filename = (
                f"performance_{optimizer}_{dataset}_{search_space}_{acc_metric}.png"
            )
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"  - Plot saved to {save_path}")

    # --- Final Plot Configuration (for combined plot) ---
    if combine_plots:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        # Get existing handles and labels from the plot (these are the mean lines)
        handles, labels = ax.get_legend_handles_labels()

        # Create a single, representative legend entry for standard deviation
        std_dev_handle = Patch(facecolor="gray", alpha=0.4, label="Std. Dev.")

        # Create custom legend handles for each seed's marker, but in gray
        seed_handles = []
        seed_labels = []
        # Use the global seed_to_marker map for a complete legend
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

        # Combine handles and create the legend in the desired order
        ax.legend(
            handles=handles + [std_dev_handle] + seed_handles,
            labels=labels + ["Std. Dev."] + seed_labels,
            loc="upper left",
            ncol=1,  # vertical
        )

        ax.set_xlabel("Runtime (Seconds) [Log Scale]")
        if acc_metric == "valid_acc":
            ax.set_ylabel("Validation Accuracy (Percent) [Linear Scale]")
        else:
            ax.set_ylabel("Training Accuracy (Percent) [Linear Scale]")
        ax.set_title(plot_title)
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, which="both", ls="-", alpha=0.5)

        filename = f"combined_performance_plot_{acc_metric}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"\nCombined plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate anytime performance plots from NASLib experiment directories."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        help="The root directory containing the experiment runs (e.g., 'testv3/').",
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
        default="plots",
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--combine_plots",
        action="store_true",
        help="If set, combines all optimizer results into a single plot.",
    )
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
