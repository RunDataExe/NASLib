import os
import json
import argparse
from collections import defaultdict

import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors


# Prefer NASLib helpers if available
try:
    from naslib.utils import get_dataset_api, get_project_root
except Exception:
    get_dataset_api = None
    get_project_root = None


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data, pretty=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2, sort_keys=False)
        else:
            json.dump(data, f)


def _resolve_nb201_api(dataset_api_obj):
    # Unwrap potential wrappers
    api_like = dataset_api_obj
    for attr in ["nb201_api", "api", "nb201", "raw_api", "_api"]:
        if hasattr(dataset_api_obj, attr):
            api_like = getattr(dataset_api_obj, attr)
            break

    # Determine number of models
    n = None
    try:
        n = len(api_like)
    except Exception:
        pass
    if n is None:
        for cand in ["n_models", "n_archs", "num", "N", "n_architectures"]:
            if hasattr(api_like, cand):
                n = int(getattr(api_like, cand))
                break
    if n is None:
        n = 15625
    return api_like, n


def _query_meta_info(api_like, idx):
    # Try common NB201 methods
    for method in ["query_meta_info_by_index", "query_by_index", "arch", "get_arch"]:
        if hasattr(api_like, method):
            fn = getattr(api_like, method)
            try:
                return fn(idx, hp="200")
            except TypeError:
                return fn(idx)
    raise RuntimeError("Could not get NB201 meta_info by index.")


def _extract_acc(meta_info, dataset, split):
    """
    Extract accuracy for split in {'valid','train'} robustly.
    """
    modes_by_split = {
        "valid": ["x-valid", "valid", "val"],
        "train": ["train", "x-train"],
    }
    # Official NB201 pattern
    for mode in modes_by_split.get(split, []):
        try:
            metrics = meta_info.get_metrics(dataset, mode)
            if isinstance(metrics, dict):
                for key in ["accuracy", "acc", "top1", "best_acc", "best", "value"]:
                    if key in metrics:
                        return float(metrics[key])
                for subkey in ["eval", "metric", "stats"]:
                    if subkey in metrics and isinstance(metrics[subkey], dict):
                        for key in ["accuracy", "acc", "top1", "best_acc", "best"]:
                            if key in metrics[subkey]:
                                return float(metrics[subkey][key])
        except Exception:
            pass

    # Fallbacks
    for attr in ["metrics", "_metrics", "data", "results"]:
        if hasattr(meta_info, attr):
            try:
                obj = getattr(meta_info, attr)
                if isinstance(obj, dict) and dataset in obj:
                    ds_metrics = obj[dataset]
                    for key in [
                        f"{m}/accuracy" for m in modes_by_split.get(split, [])
                    ] + ["accuracy", "acc", "top1"]:
                        if key in ds_metrics:
                            return float(ds_metrics[key])
            except Exception:
                pass

    # Very last resort
    for cand in [
        f"{split}_acc",
        f"{split}acc",
        f"{'x_valid' if split == 'valid' else 'x_train'}_acc",
        "acc",
    ]:
        if hasattr(meta_info, cand):
            try:
                return float(getattr(meta_info, cand))
            except Exception:
                pass

    raise RuntimeError(f"Could not extract {split} accuracy for dataset={dataset}")


def build_nb201_distribution(dataset, out_dir, split):
    """
    split in {'valid','train'}
    """
    if get_dataset_api is None:
        raise RuntimeError(
            "NASLib get_dataset_api is not available in this environment."
        )

    api = get_dataset_api("nasbench201", dataset)

    # Fallback path: NASLib-style dict with 'nb201_data'
    if isinstance(api, dict) and "nb201_data" in api:
        ds_key = "cifar10-valid" if dataset in ["cifar10", "cifar10-valid"] else dataset
        rec_key = "eval_acc1es" if split == "valid" else "train_acc1es"

        accs, indices = [], []
        for idx, (_, arch_rec) in enumerate(api["nb201_data"].items()):
            try:
                series = arch_rec[ds_key][rec_key]
                # Use last epoch value
                if isinstance(series, (list, tuple)) and len(series) > 0:
                    acc = float(series[-1])
                else:
                    acc = float(series)
                indices.append(idx)
                accs.append(acc)
            except Exception:
                continue

        order = sorted(range(len(accs)), key=lambda i: accs[i])
        accs_sorted = [accs[i] for i in order]
        indices_sorted = [indices[i] for i in order]

        data = {
            "dataset": dataset,
            "split": split,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(accs_sorted),
            "indices_sorted": indices_sorted,
            "accs_sorted": accs_sorted,
            "accuracy_type": f"{split}_top1_percent",
        }

        save_path = os.path.join(out_dir, f"nb201_{split}_acc_rank_{dataset}.json")
        save_json(save_path, data, pretty=False)
        if data["total"] == 0:
            print(
                f"[WARN] Built empty NB201 distribution for {dataset}/{split} from nb201_data."
            )
        else:
            print(
                f"[INFO] Built NB201 distribution for {dataset}/{split}: {data['total']} entries."
            )
        return data, save_path

    # Original NATS-Bench API path
    api_like, n = _resolve_nb201_api(api)
    accs = []
    indices = []
    success, fail = 0, 0
    for idx in range(n):
        try:
            meta = _query_meta_info(api_like, idx)
            acc = _extract_acc(meta, dataset, split)
            indices.append(idx)
            accs.append(acc)
            success += 1
        except Exception:
            fail += 1
            continue

    order = sorted(range(len(accs)), key=lambda i: accs[i])
    accs_sorted = [accs[i] for i in order]
    indices_sorted = [indices[i] for i in order]

    data = {
        "dataset": dataset,
        "split": split,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(accs_sorted),
        "indices_sorted": indices_sorted,
        "accs_sorted": accs_sorted,
        "accuracy_type": f"{split}_top1_percent",
    }

    save_path = os.path.join(out_dir, f"nb201_{split}_acc_rank_{dataset}.json")
    save_json(save_path, data, pretty=False)
    print(
        f"[INFO] NB201 {dataset}/{split}: success={success}, fail={fail}, saved={len(accs_sorted)}"
    )
    if data["total"] == 0:
        print(
            f"[WARN] Built empty NB201 distribution for {dataset}/{split}. Check NB201 data availability."
        )
    return data, save_path


def load_or_build_distributions(datasets, out_dir, force=False):
    dist = {}
    for ds in datasets:
        dist[ds] = {}
        for split in ["valid", "train"]:
            path = os.path.join(out_dir, f"nb201_{split}_acc_rank_{ds}.json")
            need_rebuild = force
            if not need_rebuild and os.path.exists(path):
                try:
                    cached = load_json(path)
                    # Auto-rebuild if cached file is empty/invalid
                    if (
                        not isinstance(cached, dict)
                        or cached.get("total", 0) == 0
                        or not cached.get("accs_sorted")
                    ):
                        print(
                            f"[INFO] Rebuilding empty/invalid cached distribution: {path}"
                        )
                        need_rebuild = True
                    else:
                        dist[ds][split] = cached
                        continue
                except Exception:
                    need_rebuild = True
            if need_rebuild or not os.path.exists(path):
                dist[ds][split], _ = build_nb201_distribution(ds, out_dir, split)
    return dist


def percentile_from_distribution(value, accs_sorted):
    if not accs_sorted:
        return None
    # Find the index where the value would be inserted to maintain order.
    # This corresponds to the number of architectures with lower or equal accuracy.
    pos = np.searchsorted(accs_sorted, value, side="right")
    return 100.0 * pos / float(len(accs_sorted))


def plot_final_accuracy_distribution(
    root_dir,
    dataset,
    output_dir="plots",
):
    """
    Generates a plot showing the final validation accuracy of runs
    against the NAS-Bench-201 architecture distribution for a given dataset.
    """
    print(f"Searching for runs in: {root_dir}")
    files = find_error_files(root_dir)
    if not files:
        print("No 'errors.json' files found. Exiting.")
        return

    # --- 1. Load or Build NAS-Bench-201 Distribution ---
    dist_dir = os.path.join(os.path.dirname(__file__), "nb201_distributions")
    distributions = load_or_build_distributions([dataset], dist_dir)
    if not distributions or not distributions[dataset].get("valid"):
        print(
            f"Could not load or build validation distribution for {dataset}. Exiting."
        )
        return

    val_dist = distributions[dataset]["valid"]
    accs_sorted = val_dist["accs_sorted"]
    total_archs = val_dist["total"]

    # --- 2. Group runs and process final accuracies/percentiles ---
    grouped_runs = defaultdict(list)
    for f in files:
        if f["dataset"] == dataset:
            key = (f["optimizer"], f.get("zcp_method"), f["dataset"], f["search_space"])
            grouped_runs[key].append(f)

    print(f"Found {len(grouped_runs)} experiments for dataset {dataset}.")
    os.makedirs(output_dir, exist_ok=True)

    # --- 3. Setup Plot ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the architecture distribution curve
    ax.plot(
        np.arange(total_archs),
        accs_sorted,
        color="black",
        linewidth=2,
        label="NAS-Bench-201 Architecture Distribution",
    )

    # Consistent color/marker settings
    all_seeds = sorted(list(set(f["seed"] for f in files if f["dataset"] == dataset)))
    seed_to_marker = {
        seed: MARKERS[i % len(MARKERS)] for i, seed in enumerate(all_seeds)
    }

    method_keys = sorted(set((opt, zcp) for (opt, zcp, _, _) in grouped_runs.keys()))
    opt_to_color = {
        (opt, zcp): COLORS[i % len(COLORS)] for i, (opt, zcp) in enumerate(method_keys)
    }

    # --- 4. Process and Plot each method ---
    for (optimizer, zcp_method, _, _), runs in grouped_runs.items():
        group_label = format_method_label(optimizer, zcp_method)
        color = opt_to_color[(optimizer, zcp_method)]

        final_accs = []
        final_percentiles = []
        final_ranks = []
        valid_seeds = []

        for run_meta in runs:
            try:
                with open(run_meta["path"], "r") as f:
                    data = json.load(f)

                # Use 'queried_val_acc' for final accuracy
                q_val_acc = data.get("queried_val_acc")
                if q_val_acc and isinstance(q_val_acc, list) and len(q_val_acc) > 0:
                    last_acc = q_val_acc[-1]
                    rank = np.searchsorted(accs_sorted, last_acc, side="right")
                    percentile = 100.0 * rank / float(len(accs_sorted))

                    final_accs.append(last_acc)
                    final_percentiles.append(percentile)
                    final_ranks.append(rank)
                    valid_seeds.append(run_meta["seed"])
                else:
                    print(
                        f"  - Skipping seed {run_meta['seed']} for {group_label} (no queried_val_acc)."
                    )
            except Exception as e:
                print(
                    f"  - Error processing seed {run_meta['seed']} for {group_label}: {e}"
                )

        if not final_accs:
            print(f"No valid runs found for {group_label}. Skipping.")
            continue

        # Calculate stats
        mean_percentile = np.mean(final_percentiles)
        std_percentile = np.std(final_percentiles)
        mean_acc = np.mean(final_accs)
        mean_rank = np.mean(final_ranks)

        # Create legend label
        legend_label = f"{group_label} | Avg. Percentile: {mean_percentile:.2f} ± {std_percentile:.2f}"

        # Plot thick horizontal and vertical lines for the average
        # Horizontal line for average accuracy
        ax.plot([0, mean_rank], [mean_acc, mean_acc], color=color, linewidth=2)
        # Vertical line for average rank
        ax.plot([mean_rank, mean_rank], [0, mean_acc], color=color, linewidth=2)

        # Add a proxy artist for the legend with the correct label and color
        from matplotlib.lines import Line2D

        ax.add_artist(Line2D([0], [0], color=color, lw=2, label=legend_label))

        # Plot markers for each seed on the distribution curve
        for i, acc in enumerate(final_accs):
            seed = valid_seeds[i]
            rank = final_ranks[i]
            marker = seed_to_marker.get(seed, "x")

            # Adjust marker size and width based on the marker type
            current_markersize = 8 if marker == "+" else 5
            current_markeredgewidth = 2 if marker == "+" else 1

            ax.plot(
                rank,
                acc,
                marker=marker,
                color=color,
                markersize=current_markersize,
                linestyle="None",
                markeredgewidth=current_markeredgewidth,
                zorder=10,  # Ensure markers are on top
            )

    # --- 5. Finalize Plot ---
    ax.set_xlabel(
        f"Queryable Architectures in NAS-Bench-201´s Search Space ordered by Validation Accuracy [Linear Scale]"
    )
    ax.set_ylabel("Validation Accuracy (%) [Linear Scale]")
    ax.set_title(
        f"Final Expanded Validation Accuracy in the Queryable Architecture Distribution on {dataset.upper()}"
    )
    ax.set_xlim(0, total_archs)
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", ls="--", alpha=0.6)

    # Custom legend for markers
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()

    seed_handles = []
    for seed, marker in sorted(seed_to_marker.items()):
        # Adjust marker size and width based on the marker type
        current_markersize = 8 if marker == "+" else 5
        current_markeredgewidth = 2 if marker == "+" else 1

        seed_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=marker,
                color="gray",
                markersize=current_markersize,
                markeredgewidth=current_markeredgewidth,
                label=f"Seed {seed}",
            )
        )

    # Combine method legend with seed marker legend
    ax.legend(handles=handles + seed_handles, loc="lower right")

    filename = f"final_accuracy_distribution_{dataset}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"\nDistribution plot saved to {save_path}")


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
                        time_grid, time[unique_indices], acc[unique_indices]
                    )

                    # Adjust marker size and width based on the marker type
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

                    # Adjust marker size and width based on the marker type
                    current_markersize = 8 if marker == "+" else 5
                    current_markeredgewidth = 2 if marker == "+" else 1

                    seed_handles.append(
                        Line2D(
                            [0],
                            [0],
                            linestyle="None",
                            marker=marker,
                            color=color,
                            markersize=current_markersize,
                            markeredgewidth=current_markeredgewidth,
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

        # Keep track for legend handles later
        all_handles = []
        all_labels = []

        # Process each optimizer group for this dataset
        for (optimizer, zcp_method, ds, search_space), runs in ds_groups:
            all_aucs = []
            all_trajectories = []
            all_valid_runs_meta = []
            group_label = format_method_label(optimizer, zcp_method)
            color = opt_to_color[(optimizer, zcp_method)]
            fmt = opt_to_fmt[(optimizer, zcp_method)]

            print(f"  Processing {group_label} on {dataset} ({search_space})...")

            # Process runs and collect valid data first
            for run_meta in runs:
                time, acc, run_auc = process_run_data(
                    run_meta["path"], acc_metric, dataset
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
                    time_grid, time[unique_indices], acc[unique_indices]
                )

                # Marker styling
                current_markersize = 8 if marker == "+" else 5
                current_markeredgewidth = 2 if marker == "+" else 1

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
                ax.plot(
                    time[1:],
                    acc[1:],
                    linestyle="None",
                    marker=marker,
                    color=seed_colors[i],
                    markersize=current_markersize,
                    markeredgewidth=current_markeredgewidth,
                    alpha=0.7,
                    label=None,
                )

            # Mean and std on common grid
            interpolated_accs = []
            for time, acc in all_trajectories:
                unique_indices = np.unique(time, return_index=True)[1]
                interp_acc = np.interp(
                    time_grid, time[unique_indices], acc[unique_indices]
                )
                interpolated_accs.append(interp_acc)

            mean_acc = np.mean(interpolated_accs, axis=0)
            std_acc = np.std(interpolated_accs, axis=0)

            mean_auc = np.mean(all_aucs)
            std_auc = np.std(all_aucs)
            print(f"    - AUC: {mean_auc:.2f} ± {std_auc:.2f}")
            for i, run_auc in enumerate(all_aucs):
                print(f"      - Seed {all_valid_runs_meta[i]['seed']}: {run_auc:.2f}")

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
            # Adjust marker size and width based on the marker type
            current_markersize = 8 if marker == "+" else 5
            current_markeredgewidth = 2 if marker == "+" else 1

            seed_handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=marker,
                    color="gray",
                    markersize=current_markersize,
                    markeredgewidth=current_markeredgewidth,
                )
            )
            seed_labels.append(f"{seed}")

        ax.legend(
            handles=handles + [std_dev_handle] + seed_handles,
            labels=labels + ["Std. Dev."] + seed_labels,
            loc="upper left",
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
        default="distribution",
        choices=["train_acc", "valid_acc", "distribution"],
        help="The accuracy metric to plot or 'distribution' for the final accuracy plot.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "ImageNet16-120", "cifar10"],
        help="The dataset to generate the distribution plot for.",
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
    parser.set_defaults(combine_plots=False, show_auc_text=False, show_auc_fill=False)
    args = parser.parse_args()

    if args.metric == "distribution":
        plot_final_accuracy_distribution(
            root_dir=args.root_dir,
            dataset=args.dataset,
            output_dir=args.out_dir,
        )
    else:
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
