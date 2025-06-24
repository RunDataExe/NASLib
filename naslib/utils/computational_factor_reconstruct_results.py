import os
import re
import json
import argparse
import logging
import numpy as np

# Setup a simple logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_log_file(log_path, seed):
    """
    Parses a single log.log file to extract experiment results.
    """
    try:
        with open(log_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        logging.warning(f"Log file not found: {log_path}")
        return None
    except Exception as e:
        logging.error(f"Could not read log file {log_path}: {e}")
        return None

    # Regex patterns to find the required information
    # Pattern for op_indices: e.g., [1, 0, 3, 4, 0, 0]
    op_indices_pattern = re.compile(r"Processing architecture \d+/\d+: (\[.*?\])")
    # Pattern for queried results
    queried_pattern = re.compile(
        r"Queried: TrainTime=([\d\.]+)s, TrainAcc=([\d\.]+), ValAcc=([\d\.]+), TestAcc=([\d\.]+)"
    )
    # Pattern for locally trained results
    local_train_pattern = re.compile(
        r"Local Train Finished Arch \d+: TotalTime=([\d\.]+)s, FinalTrainAcc=([\d\.]+), BestValAcc=([\d\.]+), FinalTestAcc=([\d\.]+)"
    )

    # Find matches
    op_indices_match = op_indices_pattern.search(content)
    queried_match = queried_pattern.search(content)
    local_train_match = local_train_pattern.search(content)

    if not (op_indices_match and queried_match and local_train_match):
        logging.warning(f"Could not find all required data in {log_path}. Skipping.")
        return None

    try:
        # Extract and convert data
        op_indices = json.loads(op_indices_match.group(1))

        queried_train_time = float(queried_match.group(1))
        queried_train_acc = float(queried_match.group(2))
        queried_val_acc = float(queried_match.group(3))
        queried_test_acc = float(queried_match.group(4))

        local_train_time = float(local_train_match.group(1))
        local_train_acc = float(local_train_match.group(2))
        local_val_acc = float(local_train_match.group(3))
        local_test_acc = float(local_train_match.group(4))

        # Recalculate computational factor
        computational_factor = -1.0
        if queried_train_time > 1e-6:
            computational_factor = local_train_time / queried_train_time

        # Calculate part time and its computational factor
        part_time_queried = queried_train_time * 200
        part_time_computational_factor = -1.0
        if part_time_queried > 1e-6:
            part_time_computational_factor = local_train_time / part_time_queried

        # Assemble the result dictionary
        result = {
            "op_indices": op_indices,
            "queried_train_time": queried_train_time,
            "queried_train_acc": queried_train_acc,
            "queried_val_acc": queried_val_acc,
            "queried_test_acc": queried_test_acc,
            "local_train_time": local_train_time,
            "local_train_acc": local_train_acc,
            "local_val_acc": local_val_acc,
            "local_test_acc": local_test_acc,
            "computational_factor": computational_factor,
            "part_time_queried": part_time_queried,
            "part_time_computational_factor": part_time_computational_factor,
            "seed_of_run": int(seed),
        }
        return result

    except (json.JSONDecodeError, ValueError, IndexError) as e:
        logging.error(f"Error parsing data from {log_path}: {e}")
        return None


def calculate_stats(data_list):
    """Helper to calculate summary statistics for a list of numbers."""
    if not data_list:
        return {}
    return {
        "average": float(np.mean(data_list)),
        "std_dev": float(np.std(data_list)),
        "min": float(np.min(data_list)),
        "max": float(np.max(data_list)),
    }


def main(target_dir):
    """
    Main function to find logs, parse them, and write a new results.json.
    """
    if not os.path.isdir(target_dir):
        logging.error(f"Provided path is not a directory: {target_dir}")
        return

    all_results = []

    # Iterate through subdirectories (which are named by seed)
    for subdir_name in os.listdir(target_dir):
        subdir_path = os.path.join(target_dir, subdir_name)
        if os.path.isdir(subdir_path):
            # The name of the subdirectory is the seed
            seed = subdir_name
            log_file_path = os.path.join(subdir_path, "log.log")

            if os.path.exists(log_file_path):
                logging.info(f"Processing log for seed {seed} in {subdir_path}...")
                result_data = parse_log_file(log_file_path, seed)
                if result_data:
                    all_results.append(result_data)
            else:
                logging.warning(f"No log.log found in {subdir_path}")

    if not all_results:
        logging.warning("No valid results were found. No output file will be written.")
        return

    # Sort results by seed for consistency
    all_results.sort(key=lambda x: x["seed_of_run"])

    # --- Summary for Computational Factor ---
    all_factors = [
        r["computational_factor"]
        for r in all_results
        if r.get("computational_factor", -1) > 0
    ]

    factor_summary = {}
    if all_factors:
        factor_summary = {
            "average": np.mean(all_factors),
            "std_dev": np.std(all_factors),
            "min": np.min(all_factors),
            "max": np.max(all_factors),
            "count": len(all_factors),
            "all_factors": all_factors,
        }
        logging.info("Calculated summary statistics for computational factors:")
        logging.info(f"  Average: {factor_summary['average']:.4f}")
        logging.info(f"  Std Dev: {factor_summary['std_dev']:.4f}")
        logging.info(f"  Min: {factor_summary['min']:.4f}")
        logging.info(f"  Max: {factor_summary['max']:.4f}")
        logging.info(f"  Count: {factor_summary['count']}")
    else:
        logging.warning("No valid computational factors found to calculate statistics.")

    # --- Summary for Part-Time Computational Factor ---
    all_part_time_factors = [
        r["part_time_computational_factor"]
        for r in all_results
        if r.get("part_time_computational_factor", -1) > 0
    ]

    part_time_factor_summary = {}
    if all_part_time_factors:
        part_time_factor_summary = {
            "average": np.mean(all_part_time_factors),
            "std_dev": np.std(all_part_time_factors),
            "min": np.min(all_part_time_factors),
            "max": np.max(all_part_time_factors),
            "count": len(all_part_time_factors),
            "all_factors": all_part_time_factors,
        }
        logging.info(
            "Calculated summary statistics for part-time computational factors:"
        )
        logging.info(f"  Average: {part_time_factor_summary['average']:.4f}")
        logging.info(f"  Std Dev: {part_time_factor_summary['std_dev']:.4f}")
        logging.info(f"  Min: {part_time_factor_summary['min']:.4f}")
        logging.info(f"  Max: {part_time_factor_summary['max']:.4f}")
        logging.info(f"  Count: {part_time_factor_summary['count']}")
    else:
        logging.warning(
            "No valid part-time computational factors found to calculate statistics."
        )

    # --- Summary for Queried vs. Local Metrics ---
    metrics_to_summarize = {
        "train_time": ("queried_train_time", "local_train_time"),
        "part_time": ("part_time_queried", "local_train_time"),
        "train_acc": ("queried_train_acc", "local_train_acc"),
        "val_acc": ("queried_val_acc", "local_val_acc"),
        "test_acc": ("queried_test_acc", "local_test_acc"),
    }

    metrics_summary = {}
    logging.info("Calculating summary statistics for queried vs. local metrics:")
    for metric_name, (queried_key, local_key) in metrics_to_summarize.items():
        queried_values = [r[queried_key] for r in all_results if queried_key in r]
        local_values = [r[local_key] for r in all_results if local_key in r]
        differences = [q - l for q, l in zip(queried_values, local_values)]

        queried_stats = calculate_stats(queried_values)
        local_stats = calculate_stats(local_values)
        difference_stats = calculate_stats(differences)

        metrics_summary[metric_name] = {
            "queried": queried_stats,
            "local": local_stats,
            "queried_minus_local": difference_stats,
        }

        if queried_stats and local_stats:
            logging.info(f"  Metric: {metric_name}")
            logging.info(
                f"    Queried: Avg={queried_stats['average']:.4f}, Std={queried_stats['std_dev']:.4f}, Min={queried_stats['min']:.4f}, Max={queried_stats['max']:.4f}"
            )
            logging.info(
                f"    Local:   Avg={local_stats['average']:.4f}, Std={local_stats['std_dev']:.4f}, Min={local_stats['min']:.4f}, Max={local_stats['max']:.4f}"
            )
            if difference_stats:
                logging.info(
                    f"    Diff(Q-L): Avg={difference_stats['average']:.4f}, Std={difference_stats['std_dev']:.4f}, Min={difference_stats['min']:.4f}, Max={difference_stats['max']:.4f}"
                )

    # Combine all summaries into a single object
    final_summary = {
        "computational_factor": factor_summary,
        "part_time_computational_factor": part_time_factor_summary,
        **metrics_summary,
    }

    # Combine results and summary into a single object
    final_output = {"summary": final_summary, "results": all_results}

    # Write the consolidated results to results.json in the target directory
    output_path = os.path.join(target_dir, "results.json")
    try:
        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=4)
        logging.info(f"Successfully reconstructed {len(all_results)} results.")
        logging.info(f"New results file with summary saved to: {output_path}")
    except IOError as e:
        logging.error(f"Failed to write results to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstructs results.json from log files in subdirectories."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        help="The directory containing the experiment subdirectories (e.g., .../cifar10).",
        required=True,
    )
    args = parser.parse_args()

    main(args.target_dir)
