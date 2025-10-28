#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import optuna
from optuna.trial import TrialState, FrozenTrial
import urllib.parse
import logging
import shutil
import sqlite3

# -------------------------
# Configuration/constants
# -------------------------

# Optimizers that must be time-shifted once by dataset-specific duration
ZCP_PRE_METHODS = {
    "zcp-pre_gsparsity",
    "zcp-pre_zcp_gsparsity",
}

# For robust study-name parsing
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
SEARCH_SPACES = {"nasbench201", "nasbench301"}


# -------------------------
# CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter studies by a common group budget with a one-time shift for ZCP-PRE optimizers."
    )
    p.add_argument(
        "--source",
        default="naslib/optimizers/oneshot/gsparsity/results_wide_hpo_queried_val_acc/WHPO_Databases",
        help="Path to a single SQLite DB (*.db) or a directory containing study DBs (one study per DB).",
    )
    p.add_argument(
        "--durations-dir",
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor",
        help="Directory containing arch_scores_duration_*.json files.",
    )
    p.add_argument(
        "--results-root",
        default=None,
        help="Root folder with per-trial artifacts (defaults to sibling 'WHPO' next to the DB dir).",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated list of datasets to include (e.g. cifar10,cifar100,ImageNet16-120).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be created; do not write filtered DBs.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Extra logging.",
    )
    p.add_argument(
        "--fast-prune",
        action="store_true",
        help="Much faster: copy the .db and prune non-permitted trials via raw SQLite (no per-trial re-insert).",
    )
    p.add_argument(
        "--slim-copy",
        action="store_true",
        help="When not using --fast-prune, speed up by omitting intermediate/system/user attrs in the filtered DB.",
    )
    p.add_argument(
        "--dest-dir",
        default=None,
        help="Optional directory to write filtered DBs into. Defaults to next to the source DB.",
    )
    p.add_argument(
        "--debug-timelines",
        action="store_true",
        help="Print per-trial measured+queried components, cumulative timeline, and deltas vs T_ref/effective budget.",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="Optional path to write all logs/output. If set, console + file logging are enabled.",
    )
    p.add_argument(
        "--append-log",
        action="store_true",
        help="Append to --log-file instead of overwriting.",
    )
    return p.parse_args()


# -------------------------
# Utilities
# -------------------------


def list_db_files(source: str) -> List[str]:
    if os.path.isdir(source):
        return sorted(
            [
                os.path.join(source, f)
                for f in os.listdir(source)
                if f.endswith(".db") and not f.endswith("_filtered.db")
            ]
        )
    if os.path.isfile(source) and source.endswith(".db"):
        return [source]
    raise FileNotFoundError(f"Source must be a .db file or directory: {source}")


def load_dataset_durations(durations_dir: str) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    pattern = os.path.join(durations_dir, "arch_scores_duration_*.json")
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        dataset = base.replace("arch_scores_duration_", "").replace(".json", "")
        try:
            with open(path, "r") as f:
                d = json.load(f)
            dur = float(d.get("duration", 0.0))
            mapping[dataset] = dur
        except Exception as e:
            logging.warning(f"Failed to load duration from {path}: {e}")
    return mapping


def parse_study_identity(study_name: str) -> Tuple[str, str, str, int, Optional[str]]:
    """
    Returns: optimizer, search_space, dataset, seed, zcp_method|None
    Robust parsing (optimizer and dataset may contain dashes).
    """
    parts = study_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected study name format: {study_name}")

    # strip trailing suffix until we see zcp_method or an int seed
    while parts:
        tail = parts[-1]
        if tail in ZCP_METHODS:
            break
        try:
            int(tail)
            break
        except ValueError:
            parts.pop()
    if len(parts) < 4:
        raise ValueError(f"Unexpected study name after suffix stripping: {study_name}")

    zcp_method = None
    if parts[-1] in ZCP_METHODS:
        zcp_method = parts.pop()

    seed_s = parts.pop()
    try:
        seed = int(seed_s)
    except ValueError as e:
        raise ValueError(f"Seed in study name is not an int: {study_name}") from e

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


def _make_sqlite_uri_readonly(db_abspath: str) -> str:
    quoted_path = urllib.parse.quote(db_abspath)
    return f"sqlite:///file:{quoted_path}?mode=ro&cache=shared&uri=true"


def _readonly_storage(db_path: str) -> optuna.storages.RDBStorage:
    abs_path = os.path.abspath(db_path)
    ro_uri = _make_sqlite_uri_readonly(abs_path)
    return optuna.storages.RDBStorage(
        url=ro_uri,
        engine_kwargs={"connect_args": {"uri": True}},
        skip_compatibility_check=True,
    )


def load_study(db_path: str) -> optuna.study.Study:
    """
    Assumes one study per DB, with study_name = basename(db_path) without .db
    """
    study_name = os.path.splitext(os.path.basename(db_path))[0]
    storage = _readonly_storage(db_path)
    return optuna.load_study(study_name=study_name, storage=storage)


def infer_results_root_from_db(db_path: str, override: Optional[str] = None) -> str:
    if override:
        return override
    db_dir = os.path.dirname(os.path.abspath(db_path))
    base_dir = os.path.dirname(db_dir)  # parent of WHPO_Databases
    whpo = os.path.join(base_dir, "WHPO")
    if os.path.isdir(whpo):
        return whpo
    return base_dir


def get_trial_runtime(
    results_root: str,
    optimizer_type: str,
    search_space: str,
    dataset: str,
    seed: int,
    trial_number: int,
    zcp_method: Optional[str] = None,
) -> float:
    """
    Compute total runtime for a trial from errors.json's "runtime" field (list of floats).
    """
    try:
        path_parts = [results_root, optimizer_type, search_space, dataset, str(seed)]
        # Some optimizers add an extra subfolder for zcp method (only if 'zcp_' in optimizer name)
        if "zcp_" in optimizer_type and zcp_method:
            path_parts.append(zcp_method)
        path_parts.append(f"trial_{trial_number}")
        errors_path = os.path.join(*path_parts, "errors.json")
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
        return float(sum(runtime))
    except Exception as e:
        logging.error(f"Could not read runtime for trial {trial_number}: {e}")
        return 0.0


# New helpers to compute duration up to budget + final scaled queried time
def _trial_artifact_dir(
    results_root: str,
    optimizer_type: str,
    search_space: str,
    dataset: str,
    seed: int,
    trial_number: int,
    zcp_method: Optional[str],
) -> str:
    parts = [results_root, optimizer_type, search_space, dataset, str(seed)]
    if "zcp_" in optimizer_type and zcp_method:
        parts.append(zcp_method)
    parts.append(f"trial_{trial_number}")
    return os.path.join(*parts)


def _load_errors_json(trial_dir: str) -> Optional[dict]:
    p = os.path.join(trial_dir, "errors.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to read {p}: {e}")
        return None


def _get_trial_budget(trial: FrozenTrial) -> Optional[int]:
    b = trial.user_attrs.get("budget", None)
    try:
        return int(b) if b is not None else None
    except Exception:
        return None


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


def _extract_scaled_queried_time(errors: dict, epoch_budget: int) -> float:
    """
    Extract only the final scaled queried train time for the given epoch budget.
    Accepts multiple possible shapes:
      - list of per-epoch values under 'scaled_queried_train_time'
      - scalar under 'scaled_queried_train_time'
      - dict of per-epoch values (keys as int or str epochs)
    Returns 0.0 if not found.
    """
    key = "scaled_queried_train_time"
    if errors is None:
        return 0.0

    val = errors.get(key, None)
    if val is None:
        # Try a few common variants if naming differs
        for alt in [
            "scaled_queried_train_time",
            "scaled_queried_time",
            "queried_train_time_scaled",
            "scaled_queried_train_time_per_epoch",
        ]:
            if alt in errors:
                val = errors[alt]
                break

    if val is None:
        return 0.0

    # If list of per-epoch values: pick index budget-1 (clamp to last if shorter)
    if isinstance(val, list):
        if not val:
            return 0.0
        idx = max(0, min(len(val) - 1, (epoch_budget or len(val)) - 1))
        return float(val[idx])

    # If dict keyed by epoch numbers
    if isinstance(val, dict):
        if epoch_budget is None:
            # Use largest available epoch key
            try:
                keys = sorted([int(k) for k in val.keys()])
                if not keys:
                    return 0.0
                return float(val[str(keys[-1])])
            except Exception:
                return 0.0
        # Try exact match as int or string
        if epoch_budget in val:
            return float(val[epoch_budget])
        if str(epoch_budget) in val:
            return float(val[str(epoch_budget)])
        # Fallback: use largest available
        try:
            keys = sorted([int(k) for k in val.keys()])
            if not keys:
                return 0.0
            return float(val[str(keys[-1])])
        except Exception:
            return 0.0

    # If scalar
    try:
        return float(val)
    except Exception:
        return 0.0


def compute_trial_duration_components(
    trial: FrozenTrial,
    results_root: str,
    optimizer_type: str,
    search_space: str,
    dataset: str,
    seed: int,
    zcp_method: Optional[str],
) -> Tuple[float, float, float]:
    """
    Returns (measured_sum, queried_final, total) for a trial.
    measured_sum = sum(errors['runtime'] up to epoch budget)
    queried_final = final scaled_queried train time for the budget
    total = measured_sum + queried_final
    """
    budget = _get_trial_budget(trial)

    trial_dir = _trial_artifact_dir(
        results_root,
        optimizer_type,
        search_space,
        dataset,
        seed,
        trial.number,
        zcp_method,
    )
    errors = _load_errors_json(trial_dir)

    measured_list = []
    if errors is not None:
        measured_list = errors.get("runtime", []) or []
        if not isinstance(measured_list, list):
            measured_list = []

    if budget is not None and budget > 0:
        measured_sum = float(sum(measured_list[:budget]))
    else:
        measured_sum = float(sum(measured_list))

    queried_final = _extract_scaled_queried_time(errors, budget if budget else 0)
    total = measured_sum + queried_final
    return measured_sum, queried_final, total


def compute_trial_duration(
    trial: FrozenTrial,
    results_root: str,
    optimizer_type: str,
    search_space: str,
    dataset: str,
    seed: int,
    zcp_method: Optional[str],
) -> float:
    # Reuse the components logic and return total
    _, _, total = compute_trial_duration_components(
        trial, results_root, optimizer_type, search_space, dataset, seed, zcp_method
    )
    return total


@dataclass
class StudySummary:
    db_path: str
    study_name: str
    optimizer: str
    search_space: str
    dataset: str
    seed: int
    zcp_method: Optional[str]
    total_trials: int
    finished_cleaned: int
    cumulative_runtime_s: float
    per_trial_runtime_s: List[float]
    # NEW: details for debug timelines
    per_trial_numbers: List[int]
    per_trial_measured_s: List[float]
    per_trial_queried_s: List[float]


def summarize_study(db_path: str, results_root: Optional[str]) -> StudySummary:
    study_name = os.path.splitext(os.path.basename(db_path))[0]
    optimizer, search_space, dataset, seed, zcp_method = parse_study_identity(
        study_name
    )
    if results_root is None:
        results_root = infer_results_root_from_db(db_path, None)

    study = load_study(db_path)
    cleaned = clean_trials_by_restart(study.trials)
    finished = [
        t for t in cleaned if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
    ]
    finished_sorted = sorted(
        finished, key=lambda t: t.datetime_complete or t.datetime_start or datetime.min
    )

    per_trial_runtime: List[float] = []
    per_trial_numbers: List[int] = []
    per_trial_measured: List[float] = []
    per_trial_queried: List[float] = []
    cumulative_runtime = 0.0
    for t in finished_sorted:
        msum, qfin, total = compute_trial_duration_components(
            t, results_root, optimizer, search_space, dataset, seed, zcp_method
        )
        per_trial_measured.append(msum)
        per_trial_queried.append(qfin)
        per_trial_runtime.append(total)
        per_trial_numbers.append(t.number)
        cumulative_runtime += total

    return StudySummary(
        db_path=db_path,
        study_name=study_name,
        optimizer=optimizer,
        search_space=search_space,
        dataset=dataset,
        seed=seed,
        zcp_method=zcp_method,
        total_trials=len(study.trials),
        finished_cleaned=len(finished_sorted),
        cumulative_runtime_s=cumulative_runtime,
        per_trial_runtime_s=per_trial_runtime,
        per_trial_numbers=per_trial_numbers,
        per_trial_measured_s=per_trial_measured,
        per_trial_queried_s=per_trial_queried,
    )


def _trials_fit_within_budget(per_trial_runtime: List[float], budget_s: float) -> int:
    cum = 0.0
    count = 0
    for rt in per_trial_runtime:
        if cum + rt <= budget_s + 1e-9:  # small epsilon
            cum += rt
            count += 1
        else:
            break
    return count


def filter_and_copy_study_by_timeout(
    db_path: str,
    results_root: Optional[str],
    timeout_s: float,
    *,
    fast_prune: bool = False,
    slim_copy: bool = False,
    dest_dir: Optional[str] = None,
) -> Optional[str]:
    """
    Copy the study including only COMPLETE/PRUNED trials whose cumulative runtime fits timeout_s.
    """
    orig_name = os.path.splitext(os.path.basename(db_path))[0]
    optimizer, search_space, dataset, seed, zcp_method = parse_study_identity(orig_name)
    if results_root is None:
        results_root = infer_results_root_from_db(db_path)

    try:
        original_study = load_study(db_path)  # READ-ONLY
    except Exception as e:
        logging.error(f"Failed to load study '{orig_name}' from {db_path}. Error: {e}")
        return None

    cleaned = clean_trials_by_restart(original_study.trials)
    finished = [
        t for t in cleaned if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
    ]
    finished_sorted = sorted(
        finished, key=lambda t: t.datetime_complete or t.datetime_start or datetime.min
    )

    permitted: List[FrozenTrial] = []
    cumulative = 0.0
    for t in finished_sorted:
        # Use the same duration definition as in summarize
        rt = compute_trial_duration(
            t, results_root, optimizer, search_space, dataset, seed, zcp_method
        )
        if cumulative + rt <= timeout_s + 1e-9:
            cumulative += rt
            permitted.append(t)
        else:
            break

    if not permitted:
        logging.error(
            f"No trials fit within timeout {int(timeout_s)}s for '{orig_name}'. Skipping."
        )
        return None

    base_name = os.path.basename(db_path).replace(".db", "_filtered.db")
    filtered_db_path = (
        os.path.join(dest_dir, base_name)
        if dest_dir
        else db_path.replace(".db", "_filtered.db")
    )

    if fast_prune:
        # Fast path: copy file and delete everything not in permitted
        try:
            _fast_copy_and_prune_sqlite(
                src=db_path,
                dst=filtered_db_path,
                keep_trial_numbers=[t.number for t in permitted],
                new_study_name=f"{orig_name}-filtered",
            )
            logging.info(
                f"Created filtered study '{orig_name}-filtered' (fast-prune) with {len(permitted)} trials at {filtered_db_path}"
            )
            return filtered_db_path
        except Exception as e:
            logging.warning(
                f"Fast prune failed for {orig_name}: {e}. Falling back to API copy."
            )
            if os.path.exists(filtered_db_path):
                try:
                    os.remove(filtered_db_path)
                except Exception:
                    pass

    filtered_storage = f"sqlite:///{filtered_db_path}"
    filtered_study_name = f"{orig_name}-filtered"

    filtered_study = optuna.create_study(
        storage=filtered_storage,
        study_name=filtered_study_name,
        direction=original_study.direction,
    )

    for t in permitted:
        # Slim copy optionally skips heavy blobs for speed
        filtered_study.add_trial(
            optuna.trial.create_trial(
                state=t.state,
                value=t.value if t.state == TrialState.COMPLETE else None,
                params=t.params,
                distributions=t.distributions,
                user_attrs=({} if slim_copy else t.user_attrs),
                system_attrs=({} if slim_copy else t.system_attrs),
                intermediate_values=({} if slim_copy else t.intermediate_values),
            )
        )

    logging.info(
        f"Created filtered study '{filtered_study_name}' with {len(filtered_study.trials)} trials at {filtered_db_path}"
    )
    return filtered_db_path


def _fast_copy_and_prune_sqlite(
    src: str, dst: str, keep_trial_numbers: List[int], new_study_name: str
) -> None:
    """
    Fast path: copy the SQLite DB file, then prune rows NOT in keep_trial_numbers
    across trials and all child tables that have a trial_id column.
    Also rename the study to `new_study_name`.
    """
    if not keep_trial_numbers:
        raise ValueError("keep_trial_numbers must be non-empty.")

    shutil.copy2(src, dst)
    conn = sqlite3.connect(dst)
    try:
        cur = conn.cursor()
        # Speed-focused pragmas (unsafe if power loss occurs; acceptable for offline filtering)
        cur.execute("PRAGMA journal_mode=OFF;")
        cur.execute("PRAGMA synchronous=OFF;")
        cur.execute("PRAGMA temp_store=MEMORY;")
        cur.execute("PRAGMA mmap_size=300000000;")
        cur.execute("PRAGMA cache_size=-200000;")
        cur.execute("PRAGMA foreign_keys=OFF;")
        conn.commit()

        cur.execute("BEGIN IMMEDIATE;")

        # Map trial number -> trial_id
        cur.execute("SELECT trial_id, number FROM trials;")
        rows = cur.fetchall()
        num2id: Dict[int, int] = {num: tid for (tid, num) in rows}
        keep_ids = [num2id[n] for n in keep_trial_numbers if n in num2id]
        if not keep_ids:
            raise RuntimeError("No matching trial_ids found to keep.")

        # Use a temp table to avoid SQLite parameter limits
        cur.execute("DROP TABLE IF EXISTS _keep_ids;")
        cur.execute("CREATE TEMP TABLE _keep_ids(id INTEGER PRIMARY KEY);")
        cur.executemany(
            "INSERT INTO _keep_ids(id) VALUES (?);", [(i,) for i in keep_ids]
        )

        # Discover all tables that have a 'trial_id' column
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        trial_child_tables: List[str] = []
        for t in tables:
            # skip sqlite internal and studies/trials
            if t.startswith("sqlite_") or t in ("studies", "trials"):
                continue
            try:
                cur.execute(f"PRAGMA table_info({t});")
                cols = [c[1] for c in cur.fetchall()]
            except sqlite3.DatabaseError:
                continue
            if "trial_id" in cols:
                trial_child_tables.append(t)

        # Delete rows not in keep set from child tables
        for t in trial_child_tables:
            cur.execute(
                f"DELETE FROM {t} WHERE trial_id NOT IN (SELECT id FROM _keep_ids);"
            )

        # Delete from trials table
        cur.execute(
            "DELETE FROM trials WHERE trial_id NOT IN (SELECT id FROM _keep_ids);"
        )

        # Rename the study to match our filtered file convention
        cur.execute("UPDATE studies SET study_name = ?;", (new_study_name,))

        cur.execute("COMMIT;")
        # Reclaim space
        cur.execute("VACUUM;")
        conn.commit()
    finally:
        conn.close()


# ...existing code...


def main():
    args = parse_args()
    # Configure logging to console and optional file
    level = logging.DEBUG if args.verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(args.log_file, mode=("a" if args.append_log else "w"))
        fh.setFormatter(fmt)
        fh.setLevel(level)
        root.addHandler(fh)
        logging.info(f"Logging to file: {args.log_file} (append={args.append_log})")

    if args.dest_dir:
        os.makedirs(args.dest_dir, exist_ok=True)

    db_files = list_db_files(args.source)
    durations_map = load_dataset_durations(args.durations_dir)
    include_datasets = None
    if args.datasets:
        include_datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}

    # Summarize all DBs
    summaries: List[StudySummary] = []
    for db in db_files:
        study_name = os.path.splitext(os.path.basename(db))[0]
        try:
            opti, ss, ds, sd, zcpm = parse_study_identity(study_name)
        except Exception as e:
            logging.warning(f"Skipping DB '{db}' (bad study name): {e}")
            continue
        if include_datasets and ds not in include_datasets:
            continue

        try:
            s = summarize_study(db, args.results_root)
        except Exception as e:
            logging.warning(f"Failed to summarize '{db}': {e}")
            continue
        summaries.append(s)

    if not summaries:
        logging.info("No studies found.")
        return

    # Group by (search_space, dataset, seed)
    from collections import defaultdict

    groups: Dict[Tuple[str, str, int], List[StudySummary]] = defaultdict(list)
    for s in summaries:
        groups[(s.search_space, s.dataset, s.seed)].append(s)

    # For each group, compute T_ref and filter each study with shift for ZCP-PRE
    logging.info(
        "Applying time-shifted filtering per group (search_space, dataset, seed):"
    )
    for key, metas in sorted(groups.items()):
        search_space, dataset, seed = key

        spans_non_pre = [
            m.cumulative_runtime_s for m in metas if m.optimizer not in ZCP_PRE_METHODS
        ]
        spans_all = [m.cumulative_runtime_s for m in metas]
        if spans_non_pre:
            # T_ref = max(spans_non_pre)
            # ref_note = "ref=max cumulative runtime among non-zcp-pre"
            # Max COMMON horizon = largest time that all non-pre cover => min of their totals
            T_ref = min(spans_non_pre)
            ref_note = (
                "ref=min cumulative runtime among non-zcp-pre (max common horizon)"
            )
        else:
            T_ref = max(spans_all)
            ref_note = "ref=max cumulative runtime among all (no non-pre found)"

        offset_sec = float(durations_map.get(dataset, 0.0))
        if dataset not in durations_map:
            logging.warning(
                f"No arch scoring duration for dataset '{dataset}'. Using 0s shift."
            )

        logging.info(
            f"- Group ({search_space}, {dataset}, seed={seed}) | budget={int(T_ref)}s [{ref_note}] | zcp-pre shift={int(offset_sec)}s"
        )

        # NEW: print per-trial timelines and deltas to T_ref and effective budget
        if args.debug_timelines:
            for m in sorted(metas, key=lambda x: x.study_name):
                eff_budget = (
                    max(0.0, T_ref - offset_sec)
                    if m.optimizer in ZCP_PRE_METHODS
                    else T_ref
                )
                tag = "PRE" if m.optimizer in ZCP_PRE_METHODS else "NON-PRE"
                logging.info(f"    timeline: {m.study_name} [{m.optimizer} | {tag}]")
                cum = 0.0
                for tn, ms, qf, tot in zip(
                    m.per_trial_numbers,
                    m.per_trial_measured_s,
                    m.per_trial_queried_s,
                    m.per_trial_runtime_s,
                ):
                    cum += tot
                    d_tref = T_ref - cum
                    d_eff = eff_budget - cum
                    over = "OVER-EFF" if d_eff < 0 else ""
                    logging.info(
                        f"        t{tn:04d}: measured={ms:.2f}s + queried={qf:.2f}s = {tot:.2f}s | cum={cum:.2f}s | ΔTref={d_tref:.2f}s | Δeff={d_eff:.2f}s {over}"
                    )

        for m in metas:
            if m.optimizer in ZCP_PRE_METHODS:
                effective_budget = max(0.0, T_ref - offset_sec)
                shift_note = "shifted"
            else:
                effective_budget = T_ref
                shift_note = "baseline"

            trials_fit = _trials_fit_within_budget(
                m.per_trial_runtime_s, effective_budget
            )
            status = f"{shift_note} budget={int(effective_budget)}s -> trials_kept={trials_fit}/{m.finished_cleaned}"

            if args.verbose:
                logging.info(
                    f"    debug: cum_runtime={int(m.cumulative_runtime_s)}s vs eff_budget={int(effective_budget)}s"
                )

            if args.dry_run:
                logging.info(f"    {m.study_name} [{m.optimizer}] :: {status}")
                continue

            out_path = filter_and_copy_study_by_timeout(
                m.db_path,
                args.results_root,
                effective_budget,
                fast_prune=args.fast_prune,
                slim_copy=args.slim_copy,
                dest_dir=args.dest_dir,
            )
            logging.info(
                f"    {m.study_name} [{m.optimizer}] :: {status} :: out={os.path.basename(out_path) if out_path else 'SKIPPED'}"
            )

    logging.info("Done.")


if __name__ == "__main__":
    main()
