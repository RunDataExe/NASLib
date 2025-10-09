import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import optuna
from optuna.trial import TrialState  # added

SLURM_LOG_DIR = "naslib/optimizers/oneshot/gsparsity/result_final_hp/slurm"

SLURM_HEADER = """#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-pai00070
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_log_dir}/{job_name}_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

set -eo pipefail

CPUS=${{SLURM_CPUS_PER_TASK:-32}}
THREADS=${{THREADS:-1}}
echo "[Slurm] CPUs-per-task=$CPUS -> per-proc threads=$THREADS"

export OMP_NUM_THREADS=${{OMP_NUM_THREADS:-$THREADS}}
export MKL_NUM_THREADS=${{MKL_NUM_THREADS:-$THREADS}}
export OPENBLAS_NUM_THREADS=${{OPENBLAS_NUM_THREADS:-$THREADS}}
export NUMEXPR_NUM_THREADS=${{NUMEXPR_NUM_THREADS:-$THREADS}}
export TORCH_NUM_THREADS=${{TORCH_NUM_THREADS:-$THREADS}}

export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
export MKL_DYNAMIC=FALSE

export PYTHONUNBUFFERED=1
ulimit -n ${{ULIMIT_NOFILE:-16384}} || true

"""

CONFIGURATOR = "naslib/optimizers/oneshot/gsparsity/configurator_hp.py"

# Default final-run seeds
DEFAULT_FINAL_SEEDS = [2092269736, 1544457859, 3788705088]

# Known ZCP method names to disambiguate study name parsing
ZCP_METHODS = {
    "synflow",
    "grad_norm",
    "fisher",
    "grasp",
    "jacov",
    "snip",
    "nwot",
    "epe_nas",
    "zen",
    "flops",
    "params",
}

# Sets used to route params for two-stage methods
STAGE1_FIELDS = {
    "k",
    "num_init",
    "num_ensemble",
    "predictor_type",
    "acq_fn_type",
    "encoding_type",
    "num_candidates",
    "removal_percentage",
    "acq_fn_optimization",
    "num_arches_to_mutate",
    "max_mutations",
}
TOP_SEARCH_FIELDS = {
    "batch_size",
    "train_portion",
    "cutout",
    "cutout_length",
    "cutout_prob",
    "use_real_time",
    # "epochs" and "train_epochs" are special-cased below
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Slurm scripts for final runs using best Optuna trial hyperparameters."
    )
    p.add_argument(
        "--source",
        required=True,
        help="Path to a single SQLite DB file (*.db) or a directory containing study DBs.",
    )
    p.add_argument(
        "--out-scripts-dir",
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/final_hp_setting_runs",
        help="Directory to write Slurm scripts into (subfolders per optimizer are created).",
    )
    p.add_argument(
        "--results-out-dir",
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp",
        help="Value for configurator --out_dir.",
    )
    p.add_argument(
        "--dataset-subset",
        type=float,
        default=1.0,
        help="Value for configurator --dataset_subset (default 1.0 for final runs).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="True",
        choices=["True", "False"],
        help="Pass --resume to configurator (string form expected by argparse).",
    )
    p.add_argument(
        "--search-epochs",
        type=int,
        default=100000,  # very high; early stopping should terminate the run
        help="Override for configurator --search_epochs (default: 100000 so early stopping halts the run).",
    )
    p.add_argument(
        "--eval-epochs",
        type=int,
        default=None,
        help="Optional override for configurator --eval_epochs.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds for final jobs. Defaults to 2092269736,1544457859,3788705088.",
    )
    return p.parse_args()


def list_db_files(source: str) -> List[str]:
    if os.path.isdir(source):
        return sorted(
            [os.path.join(source, f) for f in os.listdir(source) if f.endswith(".db")]
        )
    if os.path.isfile(source) and source.endswith(".db"):
        return [source]
    raise FileNotFoundError(f"Source must be a .db file or directory: {source}")


def list_studies_in_db(db_path: str) -> List[optuna.study.StudySummary]:
    storage = f"sqlite:///{os.path.abspath(db_path)}"
    try:
        return optuna.study.get_all_study_summaries(storage)
    except Exception as e:
        print(f"[WARN] Cannot list studies in {db_path}: {e}", file=sys.stderr)
        return []


def open_study(storage: str, study_name: str) -> optuna.study.Study:
    return optuna.load_study(study_name=study_name, storage=storage)


# --- New helpers for selecting trials including PRUNED while respecting highest budget and internal_early_stopped ---


def _trial_budget(t: optuna.trial.FrozenTrial) -> int:
    """
    Return the intended budget for a trial.
    Prefers user_attrs['budget'] set by the configurator; otherwise falls back
    to the last reported step (or 0 if unknown).
    """
    b = t.user_attrs.get("budget")
    if isinstance(b, (int, float)):
        return int(b)
    # fallback: last reported step
    if t.last_step is not None:
        return int(t.last_step)
    if t.intermediate_values:
        return int(max(t.intermediate_values.keys()))
    return 0


def _trial_score(t: optuna.trial.FrozenTrial) -> Optional[float]:
    """
    For COMPLETE trials: use final value.
    For PRUNED trials: use the last reported intermediate value.
    Returns None if no comparable score is available.
    """
    if t.state == TrialState.COMPLETE:
        return t.value
    if t.state == TrialState.PRUNED:
        if t.last_step is not None and t.last_step in t.intermediate_values:
            return t.intermediate_values[t.last_step]
        # fallback if last_step missing but intermediate values exist
        if t.intermediate_values:
            last_step = max(t.intermediate_values.keys())
            return t.intermediate_values[last_step]
    return None


def _trial_score_highest_val_acc(t: optuna.trial.FrozenTrial) -> Optional[float]:
    """
    Returns the highest validation accuracy for a trial:
    - For COMPLETE: use t.value.
    - For PRUNED: use the last reported intermediate value.
    Returns None if not available.
    """
    if t.state == TrialState.COMPLETE:
        return t.value
    if t.state == TrialState.PRUNED:
        if t.last_step is not None and t.last_step in t.intermediate_values:
            return t.intermediate_values[t.last_step]
        if t.intermediate_values:
            last_step = max(t.intermediate_values.keys())
            return t.intermediate_values[last_step]
    return None


def select_best_trial_including_pruned(
    study: optuna.study.Study,
) -> optuna.trial.FrozenTrial:
    """
    Select the best trial considering COMPLETE and PRUNED trials:
    - Exclude trials with user_attrs['internal_early_stopped'] == True.
    - Group by budget (highest first).
    - Within the highest budget group, choose the trial with the best score.
      Score is trial.value for COMPLETE, and last intermediate value for PRUNED.
    - If no trial in a budget group has a score, fall back to next budget.
    - If nothing matches, raise ValueError.
    """
    trials = study.get_trials(
        deepcopy=False,
        states=[TrialState.COMPLETE, TrialState.PRUNED],
    )

    # Filter out internal early-stopped
    candidates = [
        t for t in trials if not t.user_attrs.get("internal_early_stopped", False)
    ]
    if not candidates:
        raise ValueError("No eligible trials (after filtering internal_early_stopped).")

    # Organize by budget, highest first
    by_budget: Dict[int, List[optuna.trial.FrozenTrial]] = {}
    for t in candidates:
        b = _trial_budget(t)
        by_budget.setdefault(b, []).append(t)

    for budget in sorted(by_budget.keys(), reverse=True):
        bucket = by_budget[budget]
        scored: List[Tuple[float, int, int, optuna.trial.FrozenTrial]] = []
        # sort key tuple: (-score, state_pref, -last_step, -trial.number)
        # where state_pref prefers COMPLETE (0) over PRUNED (1)
        for t in bucket:
            score = _trial_score(t)
            if score is None:
                continue
            state_pref = 0 if t.state == TrialState.COMPLETE else 1
            last_step = (
                int(t.last_step)
                if t.last_step is not None
                else (
                    max(t.intermediate_values.keys()) if t.intermediate_values else -1
                )
            )
            scored.append((score, -state_pref, last_step, t.number, t))

        if scored:
            # maximize score, prefer COMPLETE (via -state_pref), prefer larger last_step, prefer larger trial number
            scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
            return scored[0][4]

    # If we reach here, either no scores exist or everything was filtered out
    raise ValueError(
        "No trials with comparable scores found among COMPLETE/PRUNED candidates."
    )


def select_best_trial_highest_val_acc(
    study: optuna.study.Study,
) -> optuna.trial.FrozenTrial:
    """
    Select the trial with the highest validation accuracy (regardless of budget, trial number, or state).
    Only consider trials that do NOT have user_attrs['internal_early_stopped'] == True.
    """
    trials = study.get_trials(
        deepcopy=False,
        states=[TrialState.COMPLETE, TrialState.PRUNED],
    )
    # Filter out internal early-stopped
    candidates = [
        t for t in trials if not t.user_attrs.get("internal_early_stopped", False)
    ]
    if not candidates:
        raise ValueError("No eligible trials (after filtering internal_early_stopped).")

    # Find the trial with the highest validation accuracy
    best_trial = None
    best_score = None
    for t in candidates:
        score = _trial_score_highest_val_acc(t)
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_trial = t

    if best_trial is not None:
        return best_trial

    raise ValueError("No trials with comparable scores found among candidates.")


def parse_study_identity(study_name: str) -> Tuple[str, str, str, int, Optional[str]]:
    """
    Returns: optimizer, search_space, dataset, seed, zcp_method|None

    Robust parsing:
    - Strip trailing suffix tokens that are neither a known zcp_method nor an int (e.g., '-filtered').
    - Optional zcp_method just before the seed.
    - Dataset may contain hyphens (e.g., 'ImageNet16-120'); we recover it as the chunk
      between search_space and seed.
    - Optimizer may contain hyphens (e.g., 'zcp-pre_gsparsity').
    """
    parts = study_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected study name format: {study_name}")

    # 1) Remove trailing suffix tokens like 'filtered' until we see a zcp_method or a seed
    while parts:
        tail = parts[-1]
        # keep if it's a zcp method or an int-like seed
        if tail in ZCP_METHODS:
            break
        try:
            int(tail)
            break  # last token is the seed -> stop stripping
        except ValueError:
            parts.pop()  # strip trailing suffix
    if len(parts) < 4:
        raise ValueError(f"Unexpected study name after suffix stripping: {study_name}")

    # 2) Optional zcp method
    zcp_method = None
    if parts[-1] in ZCP_METHODS:
        zcp_method = parts.pop()

    # 3) Seed
    seed_s = parts.pop()
    try:
        seed = int(seed_s)
    except ValueError as e:
        raise ValueError(f"Seed in study name is not an int: {study_name}") from e

    # 4) Find search_space token from the remaining pieces
    SEARCH_SPACES = {"nasbench201", "nasbench301"}
    try:
        ss_idx = max(i for i, t in enumerate(parts) if t in SEARCH_SPACES)
    except ValueError as e:
        raise ValueError(f"Search space token not found in: {study_name}") from e

    search_space = parts[ss_idx]
    optimizer = "-".join(parts[:ss_idx]).strip("-")
    dataset = "-".join(parts[ss_idx + 1 :]).strip("-")

    if not optimizer or not dataset:
        raise ValueError(f"Could not parse optimizer/dataset from: {study_name}")

    return optimizer, search_space, dataset, seed, zcp_method


def serialize_value(v) -> str:
    """
    Convert Python values into strings accepted by CfgNode.merge_from_list:
    - numbers: plain
    - bools: True/False (Python literals)
    - strings: JSON-quoted
    - None: Python literal None (not JSON null)
    - lists/dicts: JSON
    """
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return str(v)
    # Preserve lists/dicts/strings as JSON
    return json.dumps(v)


def build_hp_overrides(
    optimizer: str, params: Dict[str, object]
) -> List[Tuple[str, str]]:
    """
    Map Optuna trial.params to configurator --hp dot-paths.
    Covers all tuned params from the HPO definitions.
    """
    overrides: List[Tuple[str, str]] = []

    def add(path: str, val):
        overrides.append((path, serialize_value(val)))

    two_stage = optimizer in {
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
        "self_training_inverted_bananas_gsparsity",
        "self_training_inverted_bananas_zcp_gsparsity",
    }

    for k, v in sorted(params.items()):
        if not two_stage:
            add(f"search.{k}", v)
            continue

        if k == "train_epochs":
            add("stage1.search.train_epochs", v)
        elif k in TOP_SEARCH_FIELDS:
            add(f"search.{k}", v)
        elif k in STAGE1_FIELDS:
            add(f"stage1.search.{k}", v)
        else:
            add(f"stage2.search.{k}", v)

    return overrides


def build_command(
    optimizer: str,
    search_space: str,
    dataset: str,
    seed: int,
    out_dir: str,
    dataset_subset: float,
    resume: str,
    overrides: Sequence[Tuple[str, str]],
    zcp_method: Optional[str] = None,
    search_epochs: Optional[int] = None,
    eval_epochs: Optional[int] = None,
) -> str:
    base = [
        f"python -u {CONFIGURATOR}",
        f"--optimizer {optimizer}",
        f"--search_space {search_space}",
        f"--dataset {dataset}",
        f"--seed {seed}",
        f"--resume {resume}",
        f'--out_dir "{out_dir}"',
        f"--dataset_subset {dataset_subset}",
    ]
    if zcp_method and "zcp" in optimizer:
        base.insert(2, f"--zcp_method {zcp_method}")
    if search_epochs is not None:
        base.append(f"--search_epochs {search_epochs}")
    if eval_epochs is not None:
        base.append(f"--eval_epochs {eval_epochs}")

    for key, val in overrides:
        base.append(f"--hp {key}={val}")

    return " \\\n    ".join(base) + "\n"


def write_slurm_script(
    out_dir: str,
    optimizer: str,
    dataset: str,
    seed: int,
    cmd: str,
    zcp_method: Optional[str] = None,
    results_out_dir: str = "naslib/optimizers/oneshot/gsparsity/result_final_hp",
) -> str:
    job_suffix = f"_{zcp_method}" if zcp_method else ""
    job_name = f"final_hp_{optimizer}_{dataset}_{seed}{job_suffix}"

    script_dir = os.path.join(out_dir, optimizer)
    os.makedirs(script_dir, exist_ok=True)
    # Ensure Slurm log directory exists
    os.makedirs(SLURM_LOG_DIR, exist_ok=True)

    fname = f"{dataset}{job_suffix}_{seed}.sh"
    fpath = os.path.join(script_dir, fname)

    with open(fpath, "w") as f:
        f.write(SLURM_HEADER.format(job_name=job_name, slurm_log_dir=SLURM_LOG_DIR))

        # Write directly to the persistent results directory (no local scratch)
        # so checkpoints remain visible for resume across jobs.
        # Just run the main command.
        f.write(cmd)

    return fpath


def main():
    args = parse_args()
    db_files = list_db_files(args.source)

    explicit_seeds: Optional[List[int]] = None
    if args.seeds:
        try:
            explicit_seeds = [
                int(s.strip()) for s in args.seeds.split(",") if s.strip()
            ]
        except Exception as e:
            print(f"Failed to parse --seeds: {e}", file=sys.stderr)
            sys.exit(2)

    generated: List[str] = []

    for db in db_files:
        storage = f"sqlite:///{os.path.abspath(db)}"
        summaries = list_studies_in_db(db)
        if not summaries:
            print(
                f"[WARN] Skipping DB (no studies found): {db}",
                file=sys.stderr,
            )
            continue

        for summary in summaries:
            study_name = summary.study_name
            try:
                optimizer, search_space, dataset, study_seed, zcp_method = (
                    parse_study_identity(study_name)
                )
            except Exception as e:
                print(
                    f"[WARN] Skipping study (bad name): {study_name} -> {e}",
                    file=sys.stderr,
                )
                continue

            try:
                study = open_study(storage, study_name)
                best = select_best_trial_highest_val_acc(study)  # changed here
            except Exception as e:
                print(
                    f"[WARN] Skipping study (cannot select best trial): {study_name} -> {e}",
                    file=sys.stderr,
                )
                continue

            overrides = build_hp_overrides(optimizer, best.params)

            # Use fixed seeds by default, unless overridden via --seeds
            seeds_to_emit = (
                explicit_seeds
                if explicit_seeds is not None
                else list(DEFAULT_FINAL_SEEDS)
            )

            for seed in seeds_to_emit:
                cmd = build_command(
                    optimizer=optimizer,
                    search_space=search_space,
                    dataset=dataset,
                    seed=seed,
                    out_dir=args.results_out_dir,
                    dataset_subset=args.dataset_subset,
                    resume=args.resume,
                    overrides=overrides,
                    zcp_method=zcp_method,
                    search_epochs=args.search_epochs,
                    eval_epochs=args.eval_epochs,
                )
                script_path = write_slurm_script(
                    out_dir=args.out_scripts_dir,
                    optimizer=optimizer,
                    dataset=dataset,
                    seed=seed,
                    cmd=cmd,
                    zcp_method=zcp_method,
                    results_out_dir=args.results_out_dir,
                )
                generated.append(script_path)

                if dataset == "cifar100":
                    cifar10_cmd = build_command(
                        optimizer=optimizer,
                        search_space=search_space,
                        dataset="cifar10",
                        seed=seed,
                        out_dir=args.results_out_dir,
                        dataset_subset=args.dataset_subset,
                        resume=args.resume,
                        overrides=overrides,
                        zcp_method=zcp_method,
                        search_epochs=args.search_epochs,
                        eval_epochs=args.eval_epochs,
                    )
                    cifar10_script_path = write_slurm_script(
                        out_dir=args.out_scripts_dir,
                        optimizer=optimizer,
                        dataset="cifar10",
                        seed=seed,
                        cmd=cifar10_cmd,
                        zcp_method=zcp_method,
                        results_out_dir=args.results_out_dir,
                    )
                    generated.append(cifar10_script_path)

    if generated:
        print("Generated scripts:")
        for p in generated:
            print(f"  {p}")
    else:
        print("No scripts generated.")


if __name__ == "__main__":
    main()
