#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

DATASETS = {"cifar10", "cifar100", "ImageNet16-120"}
SECONDS_IN_48H = 48 * 60 * 60
# NEW: methods that receive one-time shift
ZCP_PRE_METHODS = {
    "zcp-pre_gsparsity",
    "zcp-pre_zcp_gsparsity",
}
# Keys to trim for random_search runs
EPOCH_KEYS = [
    "train_acc",
    "train_loss",
    "valid_acc",
    "valid_loss",
    "test_acc",
    "test_loss",
    "runtime",
    "train_time",
    "queried_test_acc",
    "queried_val_acc",
    "queried_train_acc",
    "scaled_queried_train_time",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Truncate random_search errors.json to a per-dataset budget derived from non-random_search max runtime + extra seconds."
    )
    p.add_argument(
        "--root_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp",
        help="Root directory containing optimizer runs with errors.json files.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not write changes; just print what would be done.",
    )
    p.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not create .bak backups when writing changes.",
    )
    # NEW: durations for zcp-pre shift
    p.add_argument(
        "--durations_dir",
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor",
        help="Directory containing arch_scores_duration_*.json files for zcp-pre shift.",
    )
    # NEW: random_search extra time controls
    p.add_argument(
        "--rs_extra_time",
        type=float,
        default=SECONDS_IN_48H,
        help="Additional budget (in seconds) granted to random_search for all datasets unless overridden.",
    )
    p.add_argument(
        "--rs_extra_time_override",
        type=str,
        nargs="*",
        default=[],
        help="Per-dataset overrides as DATASET=SECONDS (e.g., cifar10=7200 ImageNet16-120=14400).",
    )
    return p.parse_args()


def iter_error_files(root_dir: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root_dir):
        if "errors.json" in filenames:
            yield os.path.join(dirpath, "errors.json")


def infer_dataset_from_path(path: str) -> Optional[str]:
    # Heuristic: find a path component equal to a known dataset
    parts = path.split(os.sep)
    for part in reversed(parts):  # datasets are usually closer to the end
        if part in DATASETS:
            return part
    return None


def get_method_from_path(path: str, root_dir: str) -> str:
    rel = os.path.relpath(path, root_dir)
    return rel.split(os.sep, 1)[0]


def is_random_search(path: str, root_dir: str) -> bool:
    rel = os.path.relpath(path, root_dir)
    first = rel.split(os.sep, 1)[0]
    return first == "random_search"


def load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
        return None


def compute_final_total_time(data: dict) -> Optional[float]:
    # Returns final cumulative time (search + final eval) following the same logic
    # used elsewhere in the project.
    required = [
        "queried_train_acc",
        "queried_val_acc",
        "scaled_queried_train_time",
        "runtime",
    ]
    if not all(k in data for k in required):
        return None
    try:
        train_acc = np.asarray(data["queried_train_acc"], dtype=float)
        valid_acc = np.asarray(data["queried_val_acc"], dtype=float)
        eval_time = np.asarray(data["scaled_queried_train_time"], dtype=float)
        runtime = np.asarray(data["runtime"], dtype=float)
    except Exception:
        return None

    if not (
        train_acc.size
        and train_acc.size == valid_acc.size == eval_time.size == runtime.size
    ):
        return None

    loss = np.asarray(data.get("train_loss", []), dtype=float)
    if loss.size and loss.size != runtime.size:
        # Inconsistent; ignore loss
        loss = np.asarray([])

    is_two_stage = bool(loss.size) and np.any(loss == -1) and np.any(loss != -1)
    is_rs = bool(loss.size) and np.all(loss == -1)

    if is_rs:
        # random_search uses only eval time as cumulative cost
        total_time = np.cumsum(eval_time)
    elif is_two_stage:
        stage2_idx = np.where(loss != -1)[0]
        if stage2_idx.size == 0:
            return None
        first_stage2 = stage2_idx[0]
        stage1_cost = runtime[:first_stage2].sum()
        stage2_runtime = runtime[stage2_idx]
        stage2_cum = np.cumsum(stage2_runtime)

        # align arrays with stage2 entries
        eval_time = eval_time[stage2_idx]
        total_time = stage1_cost + stage2_cum + eval_time
    else:
        cum_search = np.cumsum(runtime)
        total_time = cum_search + eval_time

    if total_time.size == 0:
        return None
    return float(total_time[-1])


def load_zcp_pre_durations(durations_dir: str) -> Dict[str, float]:
    """
    Load per-dataset one-time durations for zcp-pre methods from durations_dir.
    Expects files named arch_scores_duration_<DATASET>.json with schema: {"duration": <float>}
    """
    durations: Dict[str, float] = {}
    for ds in DATASETS:
        fname = f"arch_scores_duration_{ds}.json"
        fpath = os.path.join(durations_dir, fname)
        data = load_json(fpath)
        if data is None:
            continue
        try:
            durations[ds] = float(data.get("duration", 0.0))
        except Exception:
            pass
    return durations


def parse_rs_extra_overrides(pairs: Iterable[str]) -> Dict[str, float]:
    """
    Parse strings like ['cifar10=7200', 'ImageNet16-120=14400'] into a dict.
    """
    out: Dict[str, float] = {}
    for item in pairs or []:
        if "=" not in item:
            print(
                f"[WARN] Invalid override '{item}', expected DATASET=SECONDS",
                file=sys.stderr,
            )
            continue
        ds, sec = item.split("=", 1)
        ds = ds.strip()
        try:
            val = float(sec.strip())
        except ValueError:
            print(f"[WARN] Invalid seconds value in override '{item}'", file=sys.stderr)
            continue
        if ds not in DATASETS:
            print(
                f"[WARN] Unknown dataset '{ds}' in override '{item}'", file=sys.stderr
            )
            continue
        out[ds] = val
    return out


def compute_dataset_budgets(
    root_dir: str,
    durations_map: Dict[str, float],
    rs_extra_time_default: float,
    rs_extra_time_overrides: Dict[str, float],
) -> Dict[str, float]:
    # For each dataset, find max final cumulative time among non-random_search runs.
    max_times: Dict[str, float] = {ds: 0.0 for ds in DATASETS}
    for path in iter_error_files(root_dir):
        ds = infer_dataset_from_path(path)
        if ds is None or ds not in DATASETS:
            continue
        if is_random_search(path, root_dir):
            continue

        data = load_json(path)
        if data is None:
            continue

        t = compute_final_total_time(data)
        if t is None or not np.isfinite(t):
            continue

        # Apply one-time zcp-pre shift when applicable
        method = get_method_from_path(path, root_dir)
        if method in ZCP_PRE_METHODS:
            offset = float(durations_map.get(str(ds), 0.0))
            t = t + offset

        if t > max_times[ds]:
            max_times[ds] = t

    # Budget = max_time + extra random_search time (global default, with per-dataset overrides)
    budgets = {
        ds: (max_times[ds] + rs_extra_time_overrides.get(ds, rs_extra_time_default))
        for ds in DATASETS
    }
    return budgets


def trim_random_search_file(
    path: str, budget: float, dry_run: bool, no_backup: bool
) -> Tuple[int, int]:
    """
    Returns (kept, original_length). If file is unchanged, kept == original_length.
    """
    data = load_json(path)
    if data is None:
        return (0, 0)

    eval_time = np.asarray(data.get("scaled_queried_train_time", []), dtype=float)
    if eval_time.size == 0:
        print(
            f"[WARN] No 'scaled_queried_train_time' in {path}; skipping.",
            file=sys.stderr,
        )
        return (0, 0)

    cum = np.cumsum(eval_time)
    # Keep all epochs with cumulative eval_time <= budget
    n_keep = int(np.searchsorted(cum, budget, side="right"))
    n_orig = int(eval_time.size)

    if n_keep >= n_orig:
        # Fits within budget; nothing to do
        return (n_orig, n_orig)

    if dry_run:
        print(
            f"[DRY-RUN] Would trim {path}: keep {n_keep}/{n_orig} epochs (budget={budget:.2f}s)"
        )
        return (n_keep, n_orig)

    # Write changes
    if not no_backup:
        try:
            os.replace(path, path + ".bak")
            src_path = path + ".bak"
        except OSError:
            # Fallback: copy-like backup using read/write
            src_path = path
            try:
                with open(path, "r") as rfh, open(path + ".bak", "w") as wfh:
                    wfh.write(rfh.read())
            except OSError:
                pass  # best effort
    else:
        src_path = path

    # Reload from src (either original or .bak)
    data = load_json(src_path)
    if data is None:
        print(f"[WARN] Failed to reload {src_path}; skipping write.", file=sys.stderr)
        return (0, n_orig)

    for key in EPOCH_KEYS:
        if key in data and isinstance(data[key], list):
            # Slice to n_keep (safe even if list shorter)
            data[key] = data[key][:n_keep]

    # Persist
    try:
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"Trimmed {path}: kept {n_keep}/{n_orig} epochs (budget={budget:.2f}s)")
    except OSError as exc:
        print(f"[ERROR] Failed to write {path}: {exc}", file=sys.stderr)
        return (0, n_orig)

    return (n_keep, n_orig)


def main() -> None:
    args = parse_args()
    root = args.root_dir

    # Load zcp-pre durations and rs extra-time overrides
    durations_map = load_zcp_pre_durations(args.durations_dir)
    rs_overrides = parse_rs_extra_overrides(args.rs_extra_time_override)

    # 1) Compute per-dataset budgets from non-random_search runs
    budgets = compute_dataset_budgets(
        root_dir=root,
        durations_map=durations_map,
        rs_extra_time_default=args.rs_extra_time,
        rs_extra_time_overrides=rs_overrides,
    )
    print("Per-dataset budgets (seconds):")
    for ds in sorted(budgets):
        extra = rs_overrides.get(ds, args.rs_extra_time)
        print(
            f"  {ds}: {budgets[ds]:.2f} (max non-RS (+ zcp-pre shift if any) + extra RS={extra:.2f}s)"
        )

    # 2) Apply truncation to random_search runs per dataset
    total_trimmed = 0
    total_files = 0
    for path in iter_error_files(root):
        if not is_random_search(path, root):
            continue
        ds = infer_dataset_from_path(path)
        if ds is None or ds not in budgets:
            continue

        kept, orig = trim_random_search_file(
            path, budgets[ds], args.dry_run, args.no_backup
        )
        if orig > 0:
            total_files += 1
            if kept < orig:
                total_trimmed += 1

    print(f"Processed {total_files} random_search runs. Trimmed: {total_trimmed}.")


if __name__ == "__main__":
    main()
