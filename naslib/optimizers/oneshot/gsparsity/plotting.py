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

# Default color and style settings for plots
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
DEFAULTS = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
]
C_MAX = 10
COLORS = [*DEFAULTS[:C_MAX]] * 3
FMTS = [*["-"] * C_MAX, *["--"] * C_MAX, *[":"] * C_MAX]
MARKERS = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "D", "d", "P", "X"]


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


def process_run_data(filepath, acc_metric):
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

        # If there are no entries for stage 2, skip this run
        if len(stage2_indices) == 0:
            return None, None, None

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

    # If after processing, we have less than 2 points, we can't plot or calculate AUC.
    if len(acc) < 1:
        return None, None, None

    # The user wants to see the raw performance, not just the incumbent.
    # incumbent_acc = np.maximum.accumulate(acc)

    # Calculate AUC on the raw accuracy, handle cases with a single point.
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

    if combine_plots:
        plt.figure(figsize=(14, 8))
        ax = plt.gca()
        plot_title = (
            f"Anytime Performance Comparison\n{acc_metric.replace('_', ' ').title()}"
        )
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
        group_label = f"{optimizer}_{dataset}"
        color = COLORS[group_idx % len(COLORS)]
        fmt = FMTS[group_idx % len(FMTS)]

        print(f"\nProcessing {optimizer} on {dataset} ({search_space})...")

        # Process runs and collect valid data first
        for run_meta in runs:
            time, acc, run_auc = process_run_data(run_meta["path"], acc_metric)
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

        # Plot individual runs from the collected valid data
        if not combine_plots:
            # For individual plots, label each seed clearly
            for i, (time, acc) in enumerate(all_trajectories):
                run_meta = all_valid_runs_meta[i]
                ax.plot(
                    time,
                    acc,
                    color="grey",
                    alpha=0.7,
                    linestyle=":",
                    marker=MARKERS[i % len(MARKERS)],
                    markersize=5,
                    label=f"Seed {run_meta['seed']}",
                )
        else:
            # For combined plot, show seeds faintly with one legend entry
            for i, (time, acc) in enumerate(all_trajectories):
                ax.plot(
                    time,
                    acc,
                    color=color,
                    alpha=0.25,
                    linestyle=":",
                    linewidth=1.5,
                    label="Individual Seeds" if i == 0 else None,
                )

        # --- Aggregation and Mean Plot ---
        # Create a common time grid for interpolation
        max_time = max(t[-1] for t, a in all_trajectories) if all_trajectories else 0
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
        mean_label = f"{group_label} (Mean of {num_seeds} seeds)"
        if show_auc_text:
            mean_label += f" | AUC: {mean_auc:.2f} ± {std_auc:.2f}"

        # Plot mean and std deviation
        ax.plot(
            time_grid,
            mean_acc,
            color=color,
            linestyle=fmt,
            linewidth=2.5,
            label=mean_label,
        )
        ax.fill_between(
            time_grid,
            mean_acc - std_acc,
            mean_acc + std_acc,
            color=color,
            alpha=0.2,
            label=f"{group_label} (Std. Dev.)" if combine_plots else "Std. Dev.",
        )

        if show_auc_fill and not combine_plots:
            ax.fill_between(
                time_grid, 0, mean_acc, color=color, alpha=0.1, label="Mean AUC Area"
            )

        # --- Final Plot Configuration (for individual plots) ---
        if not combine_plots:
            ax.legend(loc="upper left")
            ax.set_xlabel("Runtime (seconds)")
            ax.set_ylabel(f"{acc_metric.replace('_', ' ').title()}")
            ax.set_title(
                f"Anytime Performance: {optimizer}\n{dataset} on {search_space}"
            )
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

            filename = f"{optimizer}_{dataset}_{search_space}_{acc_metric}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"  - Plot saved to {save_path}")

    # --- Final Plot Configuration (for combined plot) ---
    if combine_plots:
        ax.legend(loc="upper left")
        ax.set_xlabel("Runtime (seconds)")
        ax.set_ylabel(f"{acc_metric.replace('_', ' ').title()}")
        ax.set_title(plot_title)
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, which="both", ls="-", alpha=0.5)

        filename = f"combined_plot_{acc_metric}.png"
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
