import argparse
import json
import os
import optuna
from optuna.trial import TrialState, FrozenTrial
import logging
from typing import List, Optional, Tuple
import urllib.parse

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_study_name(study_name: str) -> Tuple[str, str, str, int, Optional[str]]:
    """
    Parses study name robustly even if optimizer or dataset contain dashes.

    Expected format:
      {optimizer}-{search_space}-{dataset}-{seed}[-{zcp_method}]

    search_space ∈ {nasbench201, nasbench301}
    dataset can contain dashes (e.g., ImageNet16-120).
    optimizer can contain dashes (e.g., zcp-pre_gsparsity).
    """
    parts = study_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Study name '{study_name}' is not in the expected format.")

    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except ValueError:
            return False

    # Optional zcp_method at the end (non-int)
    zcp_method = None
    core = parts
    if not is_int(parts[-1]):
        zcp_method = parts[-1]
        core = parts[:-1]

    if not is_int(core[-1]):
        raise ValueError(
            f"Study name '{study_name}' does not end with an integer seed."
        )
    seed = int(core[-1])
    core = core[:-1]

    # Locate search_space token
    known_search_spaces = {"nasbench201", "nasbench301"}
    ss_idx = None
    for i, tok in enumerate(core):
        if tok in known_search_spaces:
            ss_idx = i
            break
    if ss_idx is None:
        raise ValueError(f"Could not find search_space in '{study_name}'.")

    optimizer = "-".join(core[:ss_idx]) if ss_idx > 0 else ""
    search_space = core[ss_idx]
    dataset_tokens = core[ss_idx + 1 :]
    if not dataset_tokens:
        raise ValueError(f"Dataset part missing in '{study_name}'.")
    dataset = "-".join(dataset_tokens)

    return optimizer, search_space, dataset, seed, zcp_method


def infer_results_root_from_db(db_path: str, override: Optional[str] = None) -> str:
    """
    Try to infer the results root folder (which contains per-trial artifacts) from the DB path.
    Expected layout in configurator:
      out_dir/
        WHPO/...
        WHPO_Databases/{study}.db

    If override is provided, return that.
    """
    if override:
        return override

    db_dir = os.path.dirname(os.path.abspath(db_path))
    base_dir = os.path.dirname(db_dir)  # parent of WHPO_Databases
    # Best-guess: sibling "WHPO"
    whpo = os.path.join(base_dir, "WHPO")
    if os.path.isdir(whpo):
        return whpo

    # Fallback: try plain parent
    logging.warning(
        f"Could not locate sibling 'WHPO' next to DB dir; falling back to parent: {base_dir}"
    )
    return base_dir


def get_trial_runtime(
    results_root: str,
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
        path_parts = [results_root, optimizer_type, search_space, dataset, str(seed)]
        # Only optimizers that add a zcp subfolder in paths use 'zcp_' in their name.
        if "zcp_" in optimizer_type and zcp_method:
            path_parts.append(zcp_method)
        path_parts.append(f"trial_{trial_number}")

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
        return total_runtime

    except (FileNotFoundError, json.JSONDecodeError, IndexError, KeyError) as e:
        logging.error(f"Could not process runtime for trial {trial_number}: {e}")
        return 0.0


def clean_trials_by_restart(trials: List[FrozenTrial]) -> List[FrozenTrial]:
    """
    Select the contiguous HPO block between the last two FAIL/RUNNING trials.
    - Exclude the very last FAIL/RUNNING (that's the canceled current run).
    - If only one FAIL/RUNNING exists, keep everything strictly before it.
    - If none exist, keep all trials.
    """
    invalid = sorted(
        (t for t in trials if t.state in (TrialState.FAIL, TrialState.RUNNING)),
        key=lambda t: t.number,
    )

    if len(invalid) == 0:
        return trials

    if len(invalid) == 1:
        cutoff = invalid[-1].number
        return [t for t in trials if t.number < cutoff]

    second_last = invalid[-2].number
    last_invalid = invalid[-1].number
    return [t for t in trials if second_last < t.number < last_invalid]


def _make_sqlite_uri_readonly(db_abspath: str) -> str:
    """
    Create a SQLite URI that opens the DB in read-only mode.
    SQLAlchemy requires uri=true in the query string.
    """
    # SQLite URI form: sqlite:///file:/abs/path/to.db?mode=ro&cache=shared&uri=true
    quoted_path = urllib.parse.quote(db_abspath)
    return f"sqlite:///file:{quoted_path}?mode=ro&cache=shared&uri=true"


def _readonly_storage(db_path: str) -> optuna.storages.RDBStorage:
    """
    Build a read-only Optuna storage for SQLite so the original DB is never modified.
    """
    abs_path = os.path.abspath(db_path)
    ro_uri = _make_sqlite_uri_readonly(abs_path)
    return optuna.storages.RDBStorage(
        url=ro_uri,
        engine_kwargs={"connect_args": {"uri": True}},
        skip_compatibility_check=True,
    )


def load_study(db_path: str) -> optuna.study.Study:
    """
    Load study in READ-ONLY mode.
    """
    study_name = os.path.splitext(os.path.basename(db_path))[0]
    storage = _readonly_storage(db_path)
    return optuna.load_study(study_name=study_name, storage=storage)


def summarize_study(db_path: str, results_root: Optional[str]) -> dict:
    """
    Summarize a study: counts and cumulative runtime across CLEANED finished trials in completion order.
    Also returns the smallest non-zero cumulative runtime and how many trials that entails.
    """
    study_name = os.path.splitext(os.path.basename(db_path))[0]
    optimizer, search_space, dataset, seed, zcp_method = parse_study_name(study_name)
    if results_root is None:
        results_root = infer_results_root_from_db(db_path)

    study = load_study(db_path)

    # Cleaned history, finished trials only
    cleaned = clean_trials_by_restart(study.trials)
    finished = [
        t for t in cleaned if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
    ]

    # Sort by completion time (respect scheduler order)
    finished_sorted = sorted(
        finished, key=lambda t: t.datetime_complete or t.datetime_start
    )

    per_trial_runtime = []
    cumulative_runtime = 0.0
    for t in finished_sorted:
        rt = get_trial_runtime(
            results_root, optimizer, search_space, dataset, seed, t.number, zcp_method
        )
        per_trial_runtime.append(rt)
        cumulative_runtime += rt

    # Smallest non-zero cumulative runtime and how many trials it includes
    min_nonzero_cum = 0.0
    trials_for_min_nonzero = 0
    running = 0.0
    for idx, rt in enumerate(per_trial_runtime, start=1):
        running += rt
        if running > 0.0:
            min_nonzero_cum = running
            trials_for_min_nonzero = idx
            break

    return {
        "db_path": db_path,
        "study_name": study_name,
        "optimizer": optimizer,
        "search_space": search_space,
        "dataset": dataset,
        "seed": seed,
        "zcp_method": zcp_method,
        "results_root": results_root,
        "total_trials": len(study.trials),
        "cleaned_total_trials": len(cleaned),
        "finished_trials_cleaned": len(finished_sorted),
        "cumulative_runtime_s": cumulative_runtime,
        "min_nonzero_cum_runtime_s": min_nonzero_cum,
        "trials_at_min_nonzero": trials_for_min_nonzero,
        # Add per-trial runtimes so we can map any budget to a count later
        "per_trial_runtime_s": per_trial_runtime,
    }


def _score_pruned_trial(t: FrozenTrial) -> Optional[float]:
    """
    Derive a scalar score for a PRUNED trial.
    Uses the last reported intermediate value; falls back to the max-step value if needed.
    Returns None if no intermediate values exist.
    """
    if t.intermediate_values:
        if t.last_step is not None and t.last_step in t.intermediate_values:
            return t.intermediate_values[t.last_step]
        # Fallback: use the value at the largest reported step
        step = max(t.intermediate_values.keys())
        return t.intermediate_values[step]
    return None


def _build_importance_study_from_trials(
    trials: List[FrozenTrial], direction: optuna.study.StudyDirection
) -> optuna.study.Study:
    """
    Create an in-memory study that includes:
      - COMPLETE trials with their final value.
      - PRUNED trials converted to COMPLETE using _score_pruned_trial(t).
    This enables computing parameter importances over COMPLETE+PRUNED.
    """
    imp_study = optuna.create_study(direction=direction)
    for t in trials:
        if t.state == TrialState.COMPLETE:
            value = t.value
        elif t.state == TrialState.PRUNED:
            value = _score_pruned_trial(t)
            if value is None:
                # Skip PRUNED trials without any reported intermediate value.
                continue
        else:
            # Skip non-finished trials.
            continue

        imp_study.add_trial(
            optuna.trial.create_trial(
                state=TrialState.COMPLETE,  # mark as COMPLETE so Optuna will include it
                value=value,
                params=t.params,
                distributions=t.distributions,
                user_attrs=t.user_attrs,
                system_attrs=t.system_attrs,
                intermediate_values=t.intermediate_values,
            )
        )
    return imp_study


def filter_and_copy_study(
    db_path: str, results_root: Optional[str], timeout: int
) -> Optional[str]:
    """
    Create a filtered copy of the study DB including COMPLETED and PRUNED trials whose cumulative
    runtime stays within the provided timeout. Original DB is only read in read-only mode.
    """
    orig_name = os.path.splitext(os.path.basename(db_path))[0]
    optimizer, search_space, dataset, seed, zcp_method = parse_study_name(orig_name)
    if results_root is None:
        results_root = infer_results_root_from_db(db_path)

    try:
        original_study = load_study(db_path)  # READ-ONLY
    except Exception as e:
        logging.error(f"Failed to load study '{orig_name}' from {db_path}. Error: {e}")
        return None

    # Cleaned history and finished trials
    cleaned = clean_trials_by_restart(original_study.trials)
    finished = [
        t for t in cleaned if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
    ]
    finished_sorted = sorted(
        finished, key=lambda t: t.datetime_complete or t.datetime_start
    )

    permitted_trials: List[FrozenTrial] = []
    cumulative_runtime = 0.0

    for t in finished_sorted:
        rt = get_trial_runtime(
            results_root, optimizer, search_space, dataset, seed, t.number, zcp_method
        )
        if cumulative_runtime + rt <= timeout:
            cumulative_runtime += rt
            permitted_trials.append(t)
        else:
            break

    if not permitted_trials:
        logging.error(
            f"No trials fit within timeout {timeout}s for study '{orig_name}'. Skipping."
        )
        return None

    # Write filtered DB next to original (create/overwrite filtered copy only)
    filtered_db_path = db_path.replace(".db", "_filtered.db")
    filtered_storage = f"sqlite:///{filtered_db_path}"
    filtered_study_name = f"{orig_name}-filtered"

    if os.path.exists(filtered_db_path):
        os.remove(filtered_db_path)

    filtered_study = optuna.create_study(
        storage=filtered_storage,
        study_name=filtered_study_name,
        direction=original_study.direction,
    )

    for t in permitted_trials:
        filtered_study.add_trial(
            optuna.trial.create_trial(
                state=t.state,
                value=t.value if t.state == TrialState.COMPLETE else None,
                params=t.params,
                distributions=t.distributions,
                user_attrs=t.user_attrs,
                system_attrs=t.system_attrs,
                intermediate_values=t.intermediate_values,
            )
        )

    # Importance incl. PRUNED (converted to COMPLETE with last intermediate value)
    try:
        importance_study = _build_importance_study_from_trials(
            permitted_trials, original_study.direction
        )
        if len(importance_study.trials) > 1:
            param_importances = optuna.importance.get_param_importances(
                importance_study
            )
            logging.info(
                "Filtered (incl. pruned) parameter importances: "
                + ", ".join(
                    f"{k}={v:.4f}"
                    for k, v in sorted(
                        param_importances.items(), key=lambda x: x[1], reverse=True
                    )
                )
            )
            fig = optuna.visualization.plot_param_importances(importance_study)
            plot_path = os.path.join(
                os.path.dirname(db_path),
                f"{orig_name}_filtered_param_importances_including_pruned.html",
            )
            fig.write_html(plot_path)
            logging.info(
                f"Saved filtered param importances (including pruned) to: {plot_path}"
            )
        else:
            logging.info("Skipping importance analysis: not enough trials with values.")
    except Exception as e:
        logging.warning(
            f"Could not calculate or plot parameter importances (including pruned): {e}"
        )

    logging.info(
        f"Created filtered study '{filtered_study_name}' with {len(filtered_study.trials)} trials at {filtered_db_path}"
    )
    return filtered_db_path


def collect_db_files(db_dir: str) -> List[str]:
    """
    Collect all .db files in a directory, skipping already-filtered copies.
    """
    files = []
    for name in os.listdir(db_dir):
        if not name.endswith(".db"):
            continue
        if name.endswith("_filtered.db"):
            continue
        files.append(os.path.join(db_dir, name))
    return sorted(files)


def _global_smallest_runtime_summary(summaries: List[dict]) -> Optional[dict]:
    """
    Find the study with the smallest cumulative runtime across all studies.
    Prefer studies with >0 runtime and >0 finished trials; if none, fall back to any.
    Returns the summary dict of the best study or None.
    """
    if not summaries:
        return None
    positive = [
        s
        for s in summaries
        if s.get("finished_trials_cleaned", 0) > 0
        and s.get("cumulative_runtime_s", 0) > 0
    ]
    candidates = positive if positive else summaries
    return min(candidates, key=lambda s: s.get("cumulative_runtime_s", float("inf")))


def _trials_fit_within_budget(per_trial_runtime: List[float], budget_s: float) -> int:
    """
    Given a list of per-trial runtimes (ordered) and a budget, return how many trials fit.
    """
    cum = 0.0
    count = 0
    for rt in per_trial_runtime:
        if cum + rt <= budget_s:
            cum += rt
            count += 1
        else:
            break
    return count


def print_summary_table(summaries: List[dict]) -> None:
    if not summaries:
        print("No databases found.")
        return

    # Compute global smallest cumulative runtime across studies first
    best = _global_smallest_runtime_summary(summaries)
    global_budget = best["cumulative_runtime_s"] if best else 0.0
    best_study_name = best["study_name"] if best else "N/A"
    best_file = os.path.basename(best["db_path"]) if best else "N/A"

    print("\nSummary of studies in directory:")
    print("-" * 80)
    for s in summaries:
        # Map the global smallest runtime budget to "how many trials fit" for this study
        trials_with_global_min = _trials_fit_within_budget(
            s.get("per_trial_runtime_s", []), global_budget
        )
        print(
            f"{os.path.basename(s['db_path'])}\n"
            f"  study: {s['study_name']}\n"
            f"  optimizer: {s['optimizer']}  search_space: {s['search_space']}  dataset: {s['dataset']}  seed: {s['seed']}  zcp: {s['zcp_method']}\n"
            f"  total_trials: {s['total_trials']}  cleaned_total_trials: {s['cleaned_total_trials']}  finished_cleaned: {s['finished_trials_cleaned']}\n"
            f"  cumulative_runtime: {int(s['cumulative_runtime_s'])}s\n"
            f"  min_nonzero_cum_runtime: {int(s['min_nonzero_cum_runtime_s'])}s  trials_at_min: {s['trials_at_min_nonzero']}\n"
            f"  trials_with_global_min_budget: {trials_with_global_min}\n"
        )
    # Global smallest cumulative runtime across all studies
    if best:
        print("-" * 80)
        print("Global smallest cumulative runtime across studies:")
        print(
            f"  study: {best_study_name}  (file: {best_file})\n"
            f"  cumulative_runtime: {int(global_budget)}s  "
            f"trials_used: {best['finished_trials_cleaned']}"
        )
        print(f"  Tip: use --timeout {int(global_budget)}")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Filter Optuna study or folder of studies based on cumulative runtime."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--db_path", type=str, help="Path to a single Optuna SQLite database file."
    )
    group.add_argument(
        "--db_dir",
        type=str,
        help="Path to a directory containing Optuna SQLite database files.",
    )

    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Root folder containing per-trial artifacts (defaults to sibling 'WHPO' next to DB dir).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum cumulative runtime in seconds. If not provided for --db_dir, you will be prompted.",
    )
    parser.add_argument(
        "--per_db_prompt",
        action="store_true",
        help="Prompt for a timeout per database instead of one global timeout for all.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print summary; do not create filtered DBs.",
    )

    args = parser.parse_args()

    if args.db_path:
        if not os.path.exists(args.db_path):
            logging.error(f"Database not found at: {args.db_path}")
            return

        # Summarize
        summary = summarize_study(args.db_path, args.results_root)
        print_summary_table([summary])

        if args.dry_run:
            return

        timeout = args.timeout
        if timeout is None:
            try:
                timeout = int(
                    input("Enter timeout (seconds) to filter this DB: ").strip()
                )
            except Exception:
                logging.error("Invalid timeout input.")
                return

        filter_and_copy_study(args.db_path, args.results_root, timeout)
        return

    # Folder mode
    if not os.path.isdir(args.db_dir):
        logging.error(f"Directory not found: {args.db_dir}")
        return

    db_files = collect_db_files(args.db_dir)
    if not db_files:
        logging.error(f"No .db files found in directory: {args.db_dir}")
        return

    summaries = [summarize_study(db, args.results_root) for db in db_files]
    print_summary_table(summaries)

    if args.dry_run:
        return

    if args.per_db_prompt:
        for db in db_files:
            base = os.path.basename(db)
            try:
                timeout = args.timeout
                if timeout is None:
                    timeout = int(
                        input(f"Enter timeout (seconds) for {base}: ").strip()
                    )
                filter_and_copy_study(db, args.results_root, timeout)
            except Exception:
                logging.error(f"Invalid timeout input for {base}; skipping.")
        return
    else:
        timeout = args.timeout
        if timeout is None:
            try:
                timeout = int(
                    input(
                        "Enter a global timeout (seconds) to apply to ALL DBs: "
                    ).strip()
                )
            except Exception:
                logging.error("Invalid timeout input.")
                return
        for db in db_files:
            filter_and_copy_study(db, args.results_root, timeout)


if __name__ == "__main__":
    main()
