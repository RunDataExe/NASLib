#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import optuna
from optuna.trial import TrialState

# Methods that should be time-shifted
ZCP_PRE_METHODS = {
    "zcp-pre_gsparsity",
    "zcp-pre_zcp_gsparsity",
}

# Known ZCP method names (for study name parsing)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check if zcp-pre methods would choose different HPs when applying a dataset-specific start-time offset."
    )
    p.add_argument(
        "--source",
        default="naslib/optimizers/oneshot/gsparsity/results_wide_hpo/WHPO_Databases",
        help="Path to a single SQLite DB (*.db) or a directory containing study DBs.",
    )
    p.add_argument(
        "--durations-dir",
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor",
        help="Directory containing arch_scores_duration_*.json files.",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated list of datasets to include (e.g. cifar10,cifar100,ImageNet16-120).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra information.",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write a JSON summary of results.",
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


def parse_study_identity(study_name: str) -> Tuple[str, str, str, int, Optional[str]]:
    """
    Returns: optimizer, search_space, dataset, seed, zcp_method|None
    Robust parsing matching your generator script.
    """
    parts = study_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected study name format: {study_name}")

    # Strip trailing suffix tokens until we see a zcp_method or an int seed
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


def _trial_score_highest_val_acc(t: optuna.trial.FrozenTrial) -> Optional[float]:
    if t.state == TrialState.COMPLETE:
        return t.value
    if t.state == TrialState.PRUNED:
        if t.last_step is not None and t.last_step in t.intermediate_values:
            return t.intermediate_values[t.last_step]
        if t.intermediate_values:
            last_step = max(t.intermediate_values.keys())
            return t.intermediate_values[last_step]
    return None


def _eligible_trials_base(
    trials: List[optuna.trial.FrozenTrial],
) -> List[optuna.trial.FrozenTrial]:
    return [
        t
        for t in trials
        if t.state in (TrialState.COMPLETE, TrialState.PRUNED)
        and not t.user_attrs.get("internal_early_stopped", False)
    ]


def select_best_overall(
    trials: List[optuna.trial.FrozenTrial],
) -> Optional[optuna.trial.FrozenTrial]:
    candidates = _eligible_trials_base(trials)
    best_t = None
    best_s = None
    for t in candidates:
        s = _trial_score_highest_val_acc(t)
        if s is not None and (best_s is None or s > best_s):
            best_s = s
            best_t = t
    return best_t


def select_best_with_cutoff(
    trials: List[optuna.trial.FrozenTrial],
    cutoff: datetime,
) -> Optional[optuna.trial.FrozenTrial]:
    candidates = _eligible_trials_base(trials)
    eligible = []
    for t in candidates:
        if t.datetime_complete is None:
            # Cannot time-place this trial - skip for cutoff selection.
            continue
        if t.datetime_complete <= cutoff:
            eligible.append(t)
    if not eligible:
        return None
    best_t = None
    best_s = None
    for t in eligible:
        s = _trial_score_highest_val_acc(t)
        if s is not None and (best_s is None or s > best_s):
            best_s = s
            best_t = t
    return best_t


def study_timespan(
    trials: List[optuna.trial.FrozenTrial],
) -> Optional[Tuple[datetime, datetime]]:
    """
    Returns (first_start, last_end) across COMPLETE/PRUNED trials that have timestamps.
    """
    times_start = [t.datetime_start for t in trials if t.datetime_start is not None]
    times_end = [t.datetime_complete for t in trials if t.datetime_complete is not None]
    if not times_start or not times_end:
        return None
    return min(times_start), max(times_end)


def load_dataset_durations(durations_dir: str) -> Dict[str, float]:
    """
    Loads durations from arch_scores_duration_*.json files into a mapping:
    {dataset_name: duration_seconds}
    """
    mapping: Dict[str, float] = {}
    pattern = os.path.join(durations_dir, "arch_scores_duration_*.json")
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        # e.g., arch_scores_duration_cifar100.json -> dataset = cifar100
        dataset = base.replace("arch_scores_duration_", "").replace(".json", "")
        try:
            with open(path, "r") as f:
                d = json.load(f)
            dur = float(d.get("duration", 0.0))
            mapping[dataset] = dur
        except Exception as e:
            print(f"[WARN] Failed to load duration from {path}: {e}", file=sys.stderr)
    return mapping


@dataclass
class StudyMeta:
    db_path: str
    study_name: str
    optimizer: str
    search_space: str
    dataset: str
    seed: int
    zcp_method: Optional[str]
    first_start: Optional[datetime]
    last_end: Optional[datetime]
    span_sec: Optional[float]
    best_overall_trial: Optional[int]
    best_overall_score: Optional[float]
    best_overall_params: Optional[Dict[str, object]]
    shifted_best_trial: Optional[int]
    shifted_best_score: Optional[float]
    shifted_best_params: Optional[Dict[str, object]]
    shift_applied_sec: float
    cutoff_time: Optional[datetime]
    note: str


def main():
    args = parse_args()
    db_files = list_db_files(args.source)
    durations_map = load_dataset_durations(args.durations_dir)

    include_datasets = None
    if args.datasets:
        include_datasets = {d.strip() for d in args.datasets.split(",") if d.strip()}

    # Collect studies per group (search_space, dataset, seed)
    groups: Dict[Tuple[str, str, int], List[StudyMeta]] = {}

    for db in db_files:
        storage = f"sqlite:///{os.path.abspath(db)}"
        summaries = list_studies_in_db(db)
        if not summaries:
            print(f"[WARN] Skipping DB (no studies found): {db}", file=sys.stderr)
            continue

        for summary in summaries:
            study_name = summary.study_name
            try:
                optimizer, search_space, dataset, seed, zcp_method = (
                    parse_study_identity(study_name)
                )
            except Exception as e:
                if args.verbose:
                    print(
                        f"[WARN] Skip study (bad name): {study_name} -> {e}",
                        file=sys.stderr,
                    )
                continue

            if include_datasets and dataset not in include_datasets:
                continue

            try:
                study = open_study(storage, study_name)
            except Exception as e:
                print(
                    f"[WARN] Cannot open study {study_name} in {db}: {e}",
                    file=sys.stderr,
                )
                continue

            trials = study.get_trials(
                deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED]
            )
            ts = study_timespan(trials)
            if ts is None:
                first_start = last_end = None
                span_sec = None
            else:
                first_start, last_end = ts
                span_sec = (last_end - first_start).total_seconds()

            best_overall = select_best_overall(trials)
            best_overall_trial = best_overall.number if best_overall else None
            best_overall_score = (
                _trial_score_highest_val_acc(best_overall) if best_overall else None
            )
            best_overall_params = best_overall.params if best_overall else None

            meta = StudyMeta(
                db_path=db,
                study_name=study_name,
                optimizer=optimizer,
                search_space=search_space,
                dataset=dataset,
                seed=seed,
                zcp_method=zcp_method,
                first_start=first_start,
                last_end=last_end,
                span_sec=span_sec,
                best_overall_trial=best_overall_trial,
                best_overall_score=best_overall_score,
                best_overall_params=best_overall_params,
                shifted_best_trial=None,
                shifted_best_score=None,
                shifted_best_params=None,
                shift_applied_sec=0.0,
                cutoff_time=None,
                note="",
            )

            key = (search_space, dataset, seed)
            groups.setdefault(key, []).append(meta)

    results: List[Dict[str, object]] = []

    for key, metas in sorted(groups.items()):
        search_space, dataset, seed = key

        # Reference time horizon T_ref: max span among non-zcp-pre methods; if none, max among all.
        spans_non_pre = [
            m.span_sec
            for m in metas
            if m.span_sec and m.optimizer not in ZCP_PRE_METHODS
        ]
        spans_all = [m.span_sec for m in metas if m.span_sec]
        if spans_non_pre:
            T_ref_sec = max(spans_non_pre)
            ref_note = "ref=max span among non-zcp-pre"
        elif spans_all:
            T_ref_sec = max(spans_all)
            ref_note = "ref=max span among all (no non-pre found)"
        else:
            # No time data for this group, skip
            if args.verbose:
                print(
                    f"[INFO] No timestamped trials in group {key}, skipping.",
                    file=sys.stderr,
                )
            continue

        # Duration offset for this dataset (if available)
        offset_sec = 0.0
        # durations_map keys are dataset names as in filenames, e.g., "cifar100", "cifar10", "ImageNet16-120"
        if dataset in durations_map:
            offset_sec = float(durations_map[dataset])
        else:
            # Gracefully handle missing file for this dataset
            print(
                f"[WARN] No arch scoring duration found for dataset '{dataset}'. Assuming 0s offset.",
                file=sys.stderr,
            )

        # Work per zcp-pre study in this group
        for m in metas:
            if m.optimizer not in ZCP_PRE_METHODS:
                # Nothing to shift; still record baseline
                results.append(
                    {
                        "study_name": m.study_name,
                        "db_path": m.db_path,
                        "optimizer": m.optimizer,
                        "search_space": m.search_space,
                        "dataset": m.dataset,
                        "seed": m.seed,
                        "is_zcp_pre": False,
                        "ref_horizon_sec": T_ref_sec,
                        "ref_horizon_note": ref_note,
                        "offset_sec": 0.0,
                        "baseline_best_trial": m.best_overall_trial,
                        "baseline_best_score": m.best_overall_score,
                        "baseline_best_params": m.best_overall_params,
                        "shifted_best_trial": m.best_overall_trial,
                        "shifted_best_score": m.best_overall_score,
                        "shifted_best_params": m.best_overall_params,
                        "changed": False,
                        "note": "non zcp-pre (no shift applied)",
                    }
                )
                continue

            # Shift applies
            m.shift_applied_sec = offset_sec

            if m.first_start is None or m.span_sec is None:
                results.append(
                    {
                        "study_name": m.study_name,
                        "db_path": m.db_path,
                        "optimizer": m.optimizer,
                        "search_space": m.search_space,
                        "dataset": m.dataset,
                        "seed": m.seed,
                        "is_zcp_pre": True,
                        "ref_horizon_sec": T_ref_sec,
                        "ref_horizon_note": ref_note,
                        "offset_sec": offset_sec,
                        "baseline_best_trial": m.best_overall_trial,
                        "baseline_best_score": m.best_overall_score,
                        "baseline_best_params": m.best_overall_params,
                        "shifted_best_trial": None,
                        "shifted_best_score": None,
                        "shifted_best_params": None,
                        "changed": None,
                        "note": "no timestamps in study; cannot evaluate shift",
                    }
                )
                continue

            # Calculate cutoff: we "add that time at the start" (offset), then allow runs until we match others' time.
            # That means only trials with completion <= first_start + (T_ref - offset) are eligible.
            if offset_sec >= T_ref_sec:
                cutoff = m.first_start - timedelta(
                    seconds=1
                )  # ensures no trial is eligible
            else:
                cutoff = m.first_start + timedelta(seconds=(T_ref_sec - offset_sec))
            m.cutoff_time = cutoff

            # Re-open study to access trials without keeping all in memory earlier
            try:
                study = open_study(
                    f"sqlite:///{os.path.abspath(m.db_path)}", m.study_name
                )
            except Exception as e:
                print(
                    f"[WARN] Cannot re-open study {m.study_name}: {e}", file=sys.stderr
                )
                results.append(
                    {
                        "study_name": m.study_name,
                        "db_path": m.db_path,
                        "optimizer": m.optimizer,
                        "search_space": m.search_space,
                        "dataset": m.dataset,
                        "seed": m.seed,
                        "is_zcp_pre": True,
                        "ref_horizon_sec": T_ref_sec,
                        "ref_horizon_note": ref_note,
                        "offset_sec": offset_sec,
                        "baseline_best_trial": m.best_overall_trial,
                        "baseline_best_score": m.best_overall_score,
                        "baseline_best_params": m.best_overall_params,
                        "shifted_best_trial": None,
                        "shifted_best_score": None,
                        "shifted_best_params": None,
                        "changed": None,
                        "note": "failed to reopen study for cutoff selection",
                    }
                )
                continue

            trials = study.get_trials(
                deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED]
            )
            shifted_best = select_best_with_cutoff(trials, cutoff)
            if shifted_best is None:
                results.append(
                    {
                        "study_name": m.study_name,
                        "db_path": m.db_path,
                        "optimizer": m.optimizer,
                        "search_space": m.search_space,
                        "dataset": m.dataset,
                        "seed": m.seed,
                        "is_zcp_pre": True,
                        "ref_horizon_sec": T_ref_sec,
                        "ref_horizon_note": ref_note,
                        "offset_sec": offset_sec,
                        "baseline_best_trial": m.best_overall_trial,
                        "baseline_best_score": m.best_overall_score,
                        "baseline_best_params": m.best_overall_params,
                        "shifted_best_trial": None,
                        "shifted_best_score": None,
                        "shifted_best_params": None,
                        "changed": None,
                        "note": "no eligible trials within shifted cutoff",
                    }
                )
                continue

            shifted_params = shifted_best.params
            changed = m.best_overall_params != shifted_params

            results.append(
                {
                    "study_name": m.study_name,
                    "db_path": m.db_path,
                    "optimizer": m.optimizer,
                    "search_space": m.search_space,
                    "dataset": m.dataset,
                    "seed": m.seed,
                    "is_zcp_pre": True,
                    "ref_horizon_sec": T_ref_sec,
                    "ref_horizon_note": ref_note,
                    "offset_sec": offset_sec,
                    "baseline_best_trial": m.best_overall_trial,
                    "baseline_best_score": m.best_overall_score,
                    "baseline_best_params": m.best_overall_params,
                    "shifted_best_trial": shifted_best.number,
                    "shifted_best_score": _trial_score_highest_val_acc(shifted_best),
                    "shifted_best_params": shifted_params,
                    "changed": changed,
                    "note": "",
                }
            )

    # Print concise report
    print("Time-shift HP comparison (grouped by search_space, dataset, seed):")
    for r in results:
        if not r["is_zcp_pre"]:
            if args.verbose:
                print(
                    f"- [NON-PRE] {r['study_name']} | baseline trial #{r['baseline_best_trial']} score={r['baseline_best_score']}"
                )
            continue
        changed = r["changed"]
        status = (
            "CHANGED" if changed else ("NO-CHANGE" if changed is not None else "N/A")
        )
        print(
            f"- {r['study_name']} | offset={int(r['offset_sec'])}s | horizon={int(r['ref_horizon_sec'])}s "
            f"| baseline #{r['baseline_best_trial']} -> shifted #{r.get('shifted_best_trial')} => {status}"
        )
        if args.verbose:
            print(
                f"    baseline score={r['baseline_best_score']} params={r['baseline_best_params']}"
            )
            print(
                f"    shifted  score={r.get('shifted_best_score')} params={r.get('shifted_best_params')}"
            )
            if r.get("note"):
                print(f"    note: {r['note']} [{r['ref_horizon_note']}]")

    if args.out_json:
        try:
            with open(args.out_json, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Wrote JSON summary to {args.out_json}")
        except Exception as e:
            print(f"[WARN] Failed to write JSON summary: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
