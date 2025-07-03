import os
import json
import argparse
import logging
import numpy as np
from collections import defaultdict

# Setup a simple logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def find_and_load_results(base_dir, experiment_name):
    """
    Finds all errors.json files for a given experiment and loads the data.
    The structure is assumed to be: <base_dir>/<exp_name>/<exp_name>/<search_space>/<dataset>/<seed>/errors.json
    """
    results = defaultdict(dict)
    experiment_path = os.path.join(base_dir, experiment_name)
    if not os.path.isdir(experiment_path):
        logging.warning(f"Experiment directory not found: {experiment_path}")
        return results

    # Walk through the directory to find errors.json
    for root, _, files in os.walk(experiment_path):
        if "errors.json" in files:
            try:
                # Extract metadata from the path
                parts = root.split(os.sep)
                seed = parts[-1]
                dataset_name = parts[-2]
                search_space = parts[-3]
            except IndexError:
                logging.warning(
                    f"Could not determine metadata for path {root}. Skipping."
                )
                continue

            file_path = os.path.join(root, "errors.json")
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if dataset_name not in results:
                        results[dataset_name] = {}
                    results[dataset_name][seed] = data
                    logging.info(
                        f"Loaded results for {experiment_name}/{dataset_name} (seed: {seed})"
                    )
            except (json.JSONDecodeError, IOError) as e:
                logging.error(f"Failed to read or parse {file_path}: {e}")

    return results


def calculate_stats(data_list):
    """Helper to calculate summary statistics for a list of numbers."""
    if not data_list:
        return {
            "average": 0.0,
            "std_dev": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
        }
    return {
        "average": float(np.mean(data_list)),
        "std_dev": float(np.std(data_list)),
        "min": float(np.min(data_list)),
        "max": float(np.max(data_list)),
        "count": len(data_list),
    }


def main(args):
    """
    Main function to compare queried vs. local results and generate a summary.
    """
    # --- Safety Checks ---
    if not args.used_scaled_queried_runtimes:
        logging.error(
            "Error: You must confirm that the 'queried' runtimes have been scaled to the target environment "
            "(e.g., 12,4 workers). Please add the '--used_scaled_queried_runtimes' flag to proceed."
        )
        return

    if not args.used_real_self_training_runtime:
        logging.error(
            "Error: You must confirm that the 'self-training' experiment was run with 'use_real_time=True'. "
            "Please add the '--used_real_self_training_runtime' flag to proceed."
        )
        return

    if not args.confirm_200_epochs:
        logging.error(
            "Error: You must confirm that both experiments were run for 200 training epochs to ensure a "
            "correct comparison. Please add the '--confirm_200_epochs' flag to proceed."
        )
        return

    queried_exp_name = "inverted_bananas"
    local_exp_name = "self_training_inverted_bananas"

    # Load results for both experiments
    queried_results = find_and_load_results(args.base_dir, queried_exp_name)
    local_results = find_and_load_results(args.base_dir, local_exp_name)

    if not queried_results or not local_results:
        logging.error("Could not find results for one or both experiments. Exiting.")
        return

    common_datasets = sorted(
        list(set(queried_results.keys()) & set(local_results.keys()))
    )

    if not common_datasets:
        logging.warning("No common datasets found between the two experiments.")
        return

    metrics_to_compare = ["train_acc", "valid_acc", "test_acc", "runtime"]
    comparison_summary = {
        "per_dataset_per_epoch": {},
        "per_dataset_all_epochs": {},
        "all_datasets_all_epochs": {},
    }
    all_datasets_data = defaultdict(lambda: defaultdict(list))

    logging.info("--- Starting Comparison ---")
    for dataset in common_datasets:
        logging.info(f"Processing dataset: {dataset}")
        queried_seeds = queried_results.get(dataset, {})
        local_seeds = local_results.get(dataset, {})
        common_seeds = sorted(list(set(queried_seeds.keys()) & set(local_seeds.keys())))

        if not common_seeds:
            logging.warning(f"No common seeds for dataset {dataset}. Skipping.")
            continue

        # --- New Runtime Computational Factor Logic ---
        all_runtime_ratios = []
        for seed in common_seeds:
            queried_data = queried_seeds[seed]
            local_data = local_seeds[seed]

            queried_runtimes = queried_data.get("runtime", [])
            local_runtimes = local_data.get("runtime", [])
            num_epochs = min(len(queried_runtimes), len(local_runtimes))

            for epoch in range(num_epochs):
                q_rt = queried_runtimes[epoch]
                l_rt = local_runtimes[epoch]

                if l_rt > 1e-6:
                    # This ratio represents the scaling factor.
                    # e.g., if q_rt is 10s and l_rt is 40s, the ratio is 0.25.
                    # This means the local run would be 0.25x as fast with more workers.
                    ratio = q_rt / l_rt
                    all_runtime_ratios.append(ratio)
                else:
                    logging.warning(
                        f"[{dataset}/{seed}] Local runtime for epoch {epoch} is near zero. Skipping ratio calculation for this epoch."
                    )

        # Calculate summary statistics for the runtime factor
        runtime_factor_stats = calculate_stats(all_runtime_ratios)
        logging.info(
            f"[{dataset}] Calculated Runtime Computational Factor. "
            f"Average: {runtime_factor_stats['average']:.4f}, StdDev: {runtime_factor_stats['std_dev']:.4f}"
        )

        # Structure the output to match the desired format for loading
        comp_factor_output = {
            "summary": {"part_time_computational_factor": runtime_factor_stats},
            "details": {
                "info": "The 'part_time_computational_factor' is the average ratio of (queried_runtime / local_runtime) per epoch.",
                "individual_ratios": all_runtime_ratios,
            },
        }

        # Save the computational factor results to the 'self_training' directory
        output_dir = os.path.join(args.base_dir, local_exp_name, "nasbench201", dataset)
        os.makedirs(output_dir, exist_ok=True)
        results_json_path = os.path.join(output_dir, "results.json")
        try:
            with open(results_json_path, "w") as f:
                json.dump(comp_factor_output, f, indent=4)
            logging.info(
                f"Saved runtime computational factor for dataset '{dataset}' to: {results_json_path}"
            )
        except IOError as e:
            logging.error(
                f"Failed to write computational factor results to {results_json_path}: {e}"
            )

        # --- Original Epoch-wise Comparison Logic (for summary.json) ---
        dataset_epoch_data = defaultdict(lambda: defaultdict(list))
        for seed in common_seeds:
            queried_data = queried_seeds[seed]
            local_data = local_seeds[seed]
            for metric in metrics_to_compare:
                q_values = queried_data.get(metric, [])
                l_values = local_data.get(metric, [])
                num_epochs = min(len(q_values), len(l_values))

                for epoch in range(num_epochs):
                    dataset_epoch_data[epoch][f"queried_{metric}"].append(
                        q_values[epoch]
                    )
                    dataset_epoch_data[epoch][f"local_{metric}"].append(l_values[epoch])
                    dataset_epoch_data[epoch][f"diff_{metric}"].append(
                        q_values[epoch] - l_values[epoch]
                    )

        # 1. Per-epoch comparison for the current dataset
        dataset_per_epoch_summary = {}
        for epoch, data in sorted(dataset_epoch_data.items()):
            epoch_summary = {}
            for metric in metrics_to_compare:
                queried_key = f"queried_{metric}"
                local_key = f"local_{metric}"
                diff_key = f"diff_{metric}"
                epoch_summary[metric] = {
                    "queried": calculate_stats(data[queried_key]),
                    "local": calculate_stats(data[local_key]),
                    "queried_minus_local": calculate_stats(data[diff_key]),
                }
            dataset_per_epoch_summary[f"epoch_{epoch}"] = epoch_summary
        comparison_summary["per_dataset_per_epoch"][dataset] = dataset_per_epoch_summary

        # 2. Across-all-epochs comparison for the current dataset
        dataset_all_epochs_summary = {}
        for metric in metrics_to_compare:
            all_queried = [
                v
                for e in dataset_epoch_data
                for v in dataset_epoch_data[e][f"queried_{metric}"]
            ]
            all_local = [
                v
                for e in dataset_epoch_data
                for v in dataset_epoch_data[e][f"local_{metric}"]
            ]
            all_diffs = [
                v
                for e in dataset_epoch_data
                for v in dataset_epoch_data[e][f"diff_{metric}"]
            ]
            dataset_all_epochs_summary[metric] = {
                "queried": calculate_stats(all_queried),
                "local": calculate_stats(all_local),
                "queried_minus_local": calculate_stats(all_diffs),
            }
            # Aggregate for overall summary
            all_datasets_data[metric]["queried"].extend(all_queried)
            all_datasets_data[metric]["local"].extend(all_local)
            all_datasets_data[metric]["diff"].extend(all_diffs)
        comparison_summary["per_dataset_all_epochs"][dataset] = (
            dataset_all_epochs_summary
        )

    # 3. Across-all-epochs and all-datasets comparison
    all_datasets_summary = {}
    for metric in metrics_to_compare:
        all_datasets_summary[metric] = {
            "queried": calculate_stats(all_datasets_data[metric]["queried"]),
            "local": calculate_stats(all_datasets_data[metric]["local"]),
            "queried_minus_local": calculate_stats(all_datasets_data[metric]["diff"]),
        }
    comparison_summary["all_datasets_all_epochs"] = all_datasets_summary

    # Write the final summary to the output file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    try:
        with open(args.output_file, "w") as f:
            json.dump(comparison_summary, f, indent=4)
        logging.info(f"Comparison summary successfully saved to: {args.output_file}")
    except IOError as e:
        logging.error(f"Failed to write summary to {args.output_file}: {e}")

    logging.info("--- Comparison Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compares queried (inverted_bananas) vs. local (self_training_inverted_bananas) results from errors.json files."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="The base directory containing the experiment folders (e.g., 'naslib/optimizers/oneshot/gsparsity/submission_scripts/self_training_bananas_verification_and_computational_factor/').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSON file for the comparison summary.",
    )
    # --- Safety Checklist Arguments ---
    parser.add_argument(
        "--used_scaled_queried_runtimes",
        required=True,
        default=False,
        help="[Safety Check] Set this flag to confirm that the 'queried' experiment's runtimes are scaled (e.g., for 12,4 workers).",
    )
    parser.add_argument(
        "--used_real_self_training_runtime",
        required=True,
        default=False,
        help="[Safety Check] Set this flag to confirm that the 'self-training' experiment was run with 'use_real_time=True'.",
    )
    parser.add_argument(
        "--confirm_200_epochs",
        required=True,
        default=False,
        help="[Safety Check] Set this flag to confirm that both experiments were run for 200 epochs.",
    )
    script_args = parser.parse_args()

    main(script_args)
