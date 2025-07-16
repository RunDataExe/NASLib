import argparse
import json
import os
import shutil
import optuna
from optuna.trial import TrialState, FrozenTrial
import logging
import re

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_study_name(study_name: str):
    """
    Parses the study name to extract optimizer, search_space, dataset, seed, and zcp_method.
    Assumes format: {optimizer}-{search_space}-{dataset}-{seed} OR
                     {optimizer}-{search_space}-{dataset}-{seed}-{zcp_method}
    """
    parts = study_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Study name '{study_name}' is not in the expected format.")

    optimizer = parts[0]
    search_space = parts[1]
    dataset = parts[2]
    seed = parts[3]
    zcp_method = parts[4] if len(parts) > 4 else None

    return optimizer, search_space, dataset, int(seed), zcp_method


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
        description="Filter an Optuna study based on trial runtimes."
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the original Optuna SQLite database file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        required=True,
        help="The maximum cumulative runtime for all trials in seconds.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db_path):
        logging.error(f"Database file not found at: {args.db_path}")
        return

    # --- Infer paths and names from the db_path ---
    try:
        # Infer study name from the db filename
        original_study_name = os.path.splitext(os.path.basename(args.db_path))[0]

        # Infer HPO directory (assuming structure: .../hpo_dir/optuna_db/study.db)
        hpo_dir = os.path.dirname(os.path.dirname(args.db_path))

        # Parse the study name to get components
        optimizer, search_space, dataset, seed, zcp_method = parse_study_name(
            original_study_name
        )
        logging.info(f"Inferred HPO directory: {hpo_dir}")
        logging.info(f"Inferred study name: {original_study_name}")
        logging.info(
            f"Parsed components: optimizer={optimizer}, search_space={search_space}, "
            f"dataset={dataset}, seed={seed}, zcp_method={zcp_method}"
        )
    except (ValueError, IndexError) as e:
        logging.error(f"Could not parse study details from db_path. Error: {e}")
        logging.error(
            "Please ensure the DB file is named like '{optimizer}-{search_space}-{dataset}-{seed}.db' "
            "and located inside an 'optuna_db' subdirectory of your main HPO output folder."
        )
        return

    # Load the original study
    storage_name = f"sqlite:///{args.db_path}"
    try:
        original_study = optuna.load_study(
            study_name=original_study_name, storage=storage_name
        )
    except Exception as e:
        logging.error(
            f"Failed to load study '{original_study_name}' from {args.db_path}. Error: {e}"
        )
        return

    logging.info(
        f"Loaded original study '{original_study_name}' with {len(original_study.trials)} trials."
    )

    # Filter trials based on cumulative runtime
    permitted_trials = []
    cumulative_runtime = 0.0

    # --- MODIFICATION START ---
    # Get all trials that have finished, including both COMPLETED and PRUNED trials.
    # This is crucial because pruned trials still consumed time and resources.
    all_trials = original_study.get_trials(
        deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)
    )

    # Sort trials by their actual completion timestamp to respect the HPO scheduler's timeline
    sorted_trials = sorted(all_trials, key=lambda t: t.datetime_complete)
    logging.info(
        f"Processing {len(sorted_trials)} finished (COMPLETE or PRUNED) trials sorted by completion time."
    )
    # --- MODIFICATION END ---

    for trial in sorted_trials:
        runtime = get_trial_runtime(
            hpo_dir,
            optimizer,
            search_space,
            dataset,
            seed,
            trial.number,
            zcp_method,
        )

        # This check is now implicitly handled by the get_trials filter, but is good for safety.
        if trial.state not in [TrialState.COMPLETE, TrialState.PRUNED]:
            logging.warning(
                f"Skipping trial {trial.number} with unexpected state: {trial.state}."
            )
            continue

        if cumulative_runtime + runtime <= args.timeout:
            cumulative_runtime += runtime
            # We only add the trial to the *final* study if it was actually successful,
            # but we always count its runtime.
            if trial.state == TrialState.COMPLETE:
                permitted_trials.append(trial)
            logging.info(
                f"Trial {trial.number} (State: {trial.state}) is COUNTED. Cumulative runtime: {cumulative_runtime:.2f}s / {args.timeout}s."
            )
        else:
            logging.warning(
                f"Trial {trial.number} (completed at {trial.datetime_complete}) is EXCLUDED. Cumulative runtime would be {cumulative_runtime + runtime:.2f}s > {args.timeout}s. Stopping."
            )
            break  # Stop including trials once the cumulative timeout is exceeded

    if not permitted_trials:
        logging.error(
            "No trials fit within the specified cumulative timeout. Cannot create filtered study."
        )
        return

    # Create a new, filtered database
    filtered_db_path = args.db_path.replace(".db", "_filtered.db")
    if os.path.exists(filtered_db_path):
        logging.warning(
            f"Filtered DB at {filtered_db_path} already exists. It will be overwritten."
        )
        os.remove(filtered_db_path)

    # Create a new study in a new database file to store only the filtered trials
    filtered_storage = f"sqlite:///{filtered_db_path}"
    filtered_study_name = f"{original_study_name}-filtered"

    logging.info(f"Creating new filtered study in: {filtered_db_path}")

    filtered_study = optuna.create_study(
        storage=filtered_storage,
        study_name=filtered_study_name,
        direction=original_study.direction,
    )

    # Add the permitted trials to the new study, preserving their original values and states.
    for trial in permitted_trials:
        # Create a new trial in the new study with the same parameters and outcome
        # Only completed trials are in `permitted_trials`, so this is safe.
        filtered_study.add_trial(
            optuna.trial.create_trial(
                state=trial.state,
                value=trial.value,
                params=trial.params,
                distributions=trial.distributions,
                intermediate_values=trial.intermediate_values,
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
                hpo_dir, f"{original_study_name}_filtered_param_importances.html"
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
