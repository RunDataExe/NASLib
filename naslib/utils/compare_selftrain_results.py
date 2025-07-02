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
                logging.warning(f"Could not determine metadata for path {root}. Skipping.")
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
        return {}
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

        dataset_epoch_data = defaultdict(lambda: defaultdict(list))

        for seed in common_seeds:
            queried_data = queried_seeds[seed]
            local_data = local_seeds[seed]

            for metric in metrics_to_compare:
                q_values = queried_data.get(metric, [])
                l_values = local_data.get(metric, [])
                num_epochs = min(len(q_values), len(l_values))

                for epoch in range(num_epochs):
                    dataset_epoch_data[epoch][f"queried_{metric}"].append(q_values[epoch])
                    dataset_epoch_data[epoch][f"local_{metric}"].append(l_values[epoch])

        # 1. Per-epoch comparison for the current dataset
        dataset_per_epoch_summary = {}
        for epoch, data in sorted(dataset_epoch_data.items()):
            epoch_summary = {}
            for metric in metrics_to_compare:
                queried_key = f"queried_{metric}"
                local_key = f"local_{metric}"
                epoch_summary[metric] = {
                    "queried": calculate_stats(data[queried_key]),
                    "local": calculate_stats(data[local_key]),
                }
            dataset_per_epoch_summary[f"epoch_{epoch}"] = epoch_summary
        comparison_summary["per_dataset_per_epoch"][dataset] = dataset_per_epoch_summary

        # 2. Across-all-epochs comparison for the current dataset
        dataset_all_epochs_summary = {}
        for metric in metrics_to_compare:
            all_queried = [v for e in dataset_epoch_data for v in dataset_epoch_data[e][f"queried_{metric}"]]
            all_local = [v for e in dataset_epoch_data for v in dataset_epoch_data[e][f"local_{metric}"]]
            dataset_all_epochs_summary[metric] = {
                "queried": calculate_stats(all_queried),
                "local": calculate_stats(all_local),
            }
            # Aggregate for overall summary
            all_datasets_data[metric]["queried"].extend(all_queried)
            all_datasets_data[metric]["local"].extend(all_local)
        comparison_summary["per_dataset_all_epochs"][dataset] = dataset_all_epochs_summary

    # 3. Across-all-epochs and all-datasets comparison
    all_datasets_summary = {}
    for metric in metrics_to_compare:
        all_datasets_summary[metric] = {
            "queried": calculate_stats(all_datasets_data[metric]["queried"]),
            "local": calculate_stats(all_datasets_data[metric]["local"]),
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
        help="The base directory containing the experiment folders (e.g., 'naslib/optimizers/oneshot/gsparsity/submission_scripts/computational_factor/').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output JSON file for the comparison summary.",
    )
    script_args = parser.parse_args()

    main(script_args)
