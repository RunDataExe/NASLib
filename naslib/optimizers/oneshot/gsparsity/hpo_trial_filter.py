import argparse
import json
import os
import shutil
import optuna
from optuna.trial import TrialState, FrozenTrial
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_db_and_results_paths(
    hpo_dir: str,
    optimizer_type: str,
    search_space: str,
    dataset: str,
    seed: int,
    zcp_method: str = None,
):
    """
    Constructs the Optuna DB file path and the results directory path from arguments.
    """
    # Construct DB path
    db_dir = os.path.join(hpo_dir, "optuna_db")
    db_name = f"{optimizer_type}_{dataset}_{seed}.db"
    db_path = os.path.join(db_dir, db_name)

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Optuna DB file not found at: {db_path}")

    # Construct study name
    study_name = f"{optimizer_type}-{search_space}-{dataset}-{seed}"

    logging.info(f"Found database: {db_path}")
    logging.info(f"Target study name: {study_name}")

    # The results directory is the base HPO directory
    results_dir = hpo_dir
    return db_path, results_dir, study_name


def get_trial_runtime(
    results_dir: str,
    optimizer_type: str,
    search_space: str,
    dataset: str,
    seed: int,
    trial_number: int,
    zcp_method: str = None,
) -> float:
    """
    Calculates the total runtime for a specific trial by reading its errors.json file.
    """
    try:
        # Construct the path to the trial directory
        path_parts = [results_dir, optimizer_type]
        if "zcp" in optimizer_type and zcp_method:
            path_parts.append(zcp_method)
        path_parts.extend([search_space, dataset, str(seed), f"trial_{trial_number}"])

        trial_dir = os.path.join(*path_parts)
        errors_path = os.path.join(trial_dir, "errors.json")

        if not os.path.exists(errors_path):
            logging.warning(
                f"errors.json not found for trial {trial_number} at {errors_path}"
            )
            return 0.0

        with open(errors_path, "r") as f:
            data = json.load(f)

        runtime = data.get("runtime", [])
        if not runtime:
            return 0.0

        total_runtime = sum(runtime)
        logging.info(f"Trial {trial_number}: Found runtime {total_runtime:.2f}s")
        return total_runtime

    except (FileNotFoundError, json.JSONDecodeError, IndexError, KeyError) as e:
        logging.error(f"Could not process runtime for trial {trial_number}: {e}")
        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Filter an Optuna study based on trial runtimes and re-evaluate hyperparameter importance."
    )
    parser.add_argument(
        "hpo_dir",
        type=str,
        help="Path to the main HPO results directory (e.g., 'naslib/optimizers/oneshot/gsparsity/test_hpo').",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        required=True,
        help="Optimizer type (e.g., 'gsparsity', 'zcp_gsparsity').",
    )
    parser.add_argument(
        "--search_space",
        type=str,
        required=True,
        help="Search space (e.g., 'nasbench201').",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset (e.g., 'cifar10')."
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="The random seed of the HPO run."
    )
    parser.add_argument(
        "--zcp_method",
        type=str,
        default=None,
        help="ZCP method if applicable (e.g., 'jacov'). Required for ZCP-based optimizers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        required=True,
        help="The maximum permitted runtime for a single trial in seconds.",
    )
    args = parser.parse_args()

    if "zcp" in args.optimizer and not args.zcp_method:
        logging.error("The --zcp_method argument is required for ZCP-based optimizers.")
        return

    try:
        db_path, results_dir, study_name = get_db_and_results_paths(
            args.hpo_dir,
            args.optimizer,
            args.search_space,
            args.dataset,
            args.seed,
            args.zcp_method,
        )
    except FileNotFoundError as e:
        logging.error(e)
        return

    # Load the original study
    storage_name = f"sqlite:///{db_path}"
    try:
        original_study = optuna.load_study(
            study_name=study_name, storage=storage_name
        )
    except Exception as e:
        logging.error(
            f"Failed to load study '{study_name}' from {db_path}. Error: {e}"
        )
        return

    logging.info(
        f"Loaded original study '{study_name}' with {len(original_study.trials)} trials."
    )

    # Filter trials based on cumulative runtime
    permitted_trials = []
    cumulative_runtime = 0.0
    # Sort trials by number to process them sequentially
    sorted_trials = sorted(original_study.get_trials(deepcopy=False), key=lambda t: t.number)

    for trial in sorted_trials:
        runtime = get_trial_runtime(
            results_dir,
            args.optimizer,
            args.search_space,
            args.dataset,
            args.seed,
            trial.number,
            args.zcp_method,
        )

        if trial.state != TrialState.COMPLETE:
            logging.warning(
                f"Trial {trial.number} did not complete (State: {trial.state}). Stopping here."
            )
            break  # Stop if a trial in the sequence is not complete

        if cumulative_runtime + runtime <= args.timeout:
            cumulative_runtime += runtime
            permitted_trials.append(trial)
            logging.info(
                f"Trial {trial.number} is PERMITTED. Cumulative runtime: {cumulative_runtime:.2f}s / {args.timeout}s."
            )
        else:
            logging.warning(
                f"Trial {trial.number} is EXCLUDED. Cumulative runtime would be {cumulative_runtime + runtime:.2f}s > {args.timeout}s. Stopping."
            )
            break  # Stop including trials once the cumulative timeout is exceeded

    if not permitted_trials:
        logging.error(
            "No trials fit within the specified cumulative timeout. Cannot create filtered study."
        )
        return

    # Create a new, filtered database by copying the original
    filtered_db_path = db_path.replace(".db", "_filtered.db")
    if os.path.exists(filtered_db_path):
        logging.warning(
            f"Filtered DB at {filtered_db_path} already exists. It will be overwritten."
        )
        os.remove(filtered_db_path)

    # Create a new study in a new database file to store only the filtered trials
    filtered_storage = f"sqlite:///{filtered_db_path}"
    filtered_study_name = f"{study_name}-filtered"

    logging.info(f"Creating new filtered study in: {filtered_db_path}")

    filtered_study = optuna.create_study(
        storage=filtered_storage,
        study_name=filtered_study_name,
        direction=original_study.direction,
    )

    # Add the permitted trials to the new study, preserving their original values and states.
    for trial in permitted_trials:
        # Create a new trial in the new study with the same parameters and outcome
        filtered_study.add_trial(
            optuna.trial.create_trial(
                state=trial.state,
                value=trial.value,
                params=trial.params,
                distributions=trial.distributions,
            )
        )

    logging.info(
        f"Created new study '{filtered_study_name}' with {len(filtered_study.trials)} permitted trials."
    )

    # --- Hyperparameter Importance Analysis on Filtered Study ---
    print("\n--- Filtered Hyperparameter Importance Analysis ---")
    try:
        # Ensure there are enough completed trials in the new study
        completed_filtered_trials = filtered_study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE]
        )
        if len(completed_filtered_trials) > 1:
            param_importances = optuna.importance.get_param_importances(
                study=filtered_study
            )
            print("Parameter importances (fANOVA):")
            sorted_importances = sorted(
                param_importances.items(), key=lambda x: x[1], reverse=True
            )
            for param, importance in sorted_importances:
                print(f"  {param}: {importance:.4f}")

            fig = optuna.visualization.plot_param_importances(filtered_study)
            plot_path = os.path.join(
                args.hpo_dir, f"{study_name}_filtered_param_importances.html"
            )
            fig.write_html(plot_path)
            print(f"\nSaved filtered parameter importance plot to: {plot_path}")
        else:
            print(
                "Skipping importance analysis: not enough completed trials in the filtered study to analyze."
            )
    except Exception as e:
        print(
            f"Could not calculate or plot parameter importances on filtered study: {e}"
        )


if __name__ == "__main__":
    main()
