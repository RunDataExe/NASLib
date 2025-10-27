#!/usr/bin/env python3
import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from glob import glob

# ------------------------------------------------------------
# Helpers for parsing study identity (copied/adapted robustly)
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Time parsing
# ------------------------------------------------------------


def parse_i_timestamp(line: str) -> Optional[datetime]:
    """
    Parse lines like:
      [I 2025-10-24 14:42:56,430] ...
    """
    m = re.search(r"\[I\s+(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}),(\d{3})\]", line)
    if not m:
        return None
    ds, ts, ms = m.groups()
    try:
        base = datetime.strptime(f"{ds} {ts}", "%Y-%m-%d %H:%M:%S")
        return base.replace(microsecond=int(ms) * 1000)
    except Exception:
        return None


def parse_bracket_timestamp(line: str, fallback_year: int) -> Optional[datetime]:
    """
    Parse lines like:
      [10/24 14:43:23 nl.defaults...]: ...
    We need a year; use fallback_year from the [I ...] lines if available, else current year.
    """
    m = re.search(r"\[(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\s", line)
    if not m:
        return None
    mm, dd, hh, mi, ss = map(int, m.groups())
    try:
        return datetime(fallback_year, mm, dd, hh, mi, ss)
    except Exception:
        return None


# ------------------------------------------------------------
# Slurm/HPO parsing
# ------------------------------------------------------------


class HPOTrial:
    def __init__(self, number: int):
        self.number = number
        self.budget: Optional[int] = None
        self.state: Optional[str] = None  # "finished" or "pruned"
        self.start: Optional[datetime] = None
        self.end: Optional[datetime] = None
        self.dirpath: Optional[str] = None  # trial directory path
        # per-epoch info
        self.epoch_starts: Dict[int, datetime] = {}
        self.epoch_ends: Dict[int, datetime] = {}
        self.epoch_durations: Dict[int, float] = {}  # seconds


def detect_is_hpo(text: str) -> bool:
    return "A new study created in RDB with name:" in text or "Trial " in text


def extract_study_name(text: str) -> Optional[str]:
    m = re.search(r"A new study created in RDB with name:\s*([^\s]+)", text)
    return m.group(1).strip() if m else None


def extract_year_from_any_i_line(text: str) -> Optional[int]:
    m = re.search(r"\[I\s+(\d{4})-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\]", text)
    return int(m.group(1)) if m else None


def find_hpo_trial_events(text: str) -> Dict[int, HPOTrial]:
    """
    Collect budget and end-state timestamps for each trial; start time
    is assigned based on:
      - Trial 0 starts at 'A new study created...' timestamp
      - Trial n>0 starts at the previous trial's end timestamp
    """
    year = extract_year_from_any_i_line(text) or datetime.now().year

    # Map trial number -> HPOTrial
    trials: Dict[int, HPOTrial] = {}

    # Order events as they appear; we attach timestamps for end events
    end_events: List[Tuple[datetime, int, str]] = []  # (time, trial, state)
    budgets: Dict[int, int] = {}

    # Find "intended budget"
    for m in re.finditer(
        r"Trial\s+(\d+)\s+has an intended budget of\s+(\d+)\s+steps\.", text
    ):
        tnum = int(m.group(1))
        bud = int(m.group(2))
        budgets[tnum] = bud

    # Find end states
    for m in re.finditer(
        r"(\[I\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\].*?Trial\s+(\d+)\s+(finished|pruned))",
        text,
    ):
        full = m.group(1)
        tnum = int(m.group(2))
        state = m.group(3)
        ts = parse_i_timestamp(full)
        if ts:
            end_events.append((ts, tnum, state))

    # Anchor start
    mstart = re.search(
        r"(\[I\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\].*A new study created in RDB with name:)",
        text,
    )
    global_start = parse_i_timestamp(mstart.group(1)) if mstart else None

    # Create trials objects and fill budget/end
    for _, tnum, _ in end_events:
        if tnum not in trials:
            trials[tnum] = HPOTrial(tnum)
    for tnum, t in trials.items():
        if tnum in budgets:
            t.budget = budgets[tnum]

    # Sort end events by time and build starts
    end_events.sort(key=lambda x: x[0])
    # Map trial -> end time/state (last seen)
    by_trial_end: Dict[int, Tuple[datetime, str]] = {}
    for ts, tnum, state in end_events:
        by_trial_end[tnum] = (ts, state)

    # Build ordered list by trial number appearance in log
    ordered_trials = sorted(by_trial_end.items(), key=lambda kv: kv[1][0])

    # Assign start times per user's convention:
    # - Trial0 start = global_start
    # - Trial n>0 start = previous trial's end
    prev_end = global_start
    for idx, (tnum, (tend, state)) in enumerate(ordered_trials):
        t = trials[tnum]
        t.end = tend
        t.state = state
        t.start = prev_end
        prev_end = tend

    # Try to attach dirpath hints from "Resuming" or "Copying artifacts"
    for m in re.finditer(
        r"Found existing errors\.json at\s+([^\s]+/errors\.json)", text
    ):
        p = m.group(1).strip()
        mnum = re.search(r"/trial_(\d+)/errors\.json", p)
        if mnum:
            tnum = int(mnum.group(1))
            trial_dir = os.path.dirname(p)
            if tnum in trials:
                trials[tnum].dirpath = trial_dir

    for m in re.finditer(r"Copying artifacts from\s+[^\s]+\s+to\s+([^\s]+)", text):
        dest = m.group(1).strip()
        mnum = re.search(r"/trial_(\d+)$", dest)
        if mnum:
            tnum = int(mnum.group(1))
            if tnum in trials:
                trials[tnum].dirpath = dest

    return trials


def reconstruct_hpo_trial_dir(
    out_root: str,
    optimizer: str,
    search_space: str,
    dataset: str,
    seed: int,
    zcp_method: Optional[str],
    trial_number: int,
) -> str:
    parts = [out_root, "WHPO", optimizer, search_space, dataset, str(seed)]
    if "zcp_" in optimizer and zcp_method:
        parts.append(zcp_method)
    parts.append(f"trial_{trial_number}")
    return os.path.join(*parts)


def parse_epoch_timestamps_in_window(
    text: str,
    year_hint: int,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Tuple[Dict[int, datetime], Dict[int, datetime]]:
    """
    Within [start, end], collect epoch start/end timestamps.
    - Start marker: first 'Epoch E-' line
    - End marker: prefer 'Epoch E done.'; else 'Epoch E, Queried expanded anytime results'
    """
    epoch_starts: Dict[int, datetime] = {}
    epoch_ends: Dict[int, datetime] = {}
    lines = text.splitlines()

    rex_start = re.compile(r"Epoch\s+(\d+)-\d+")
    rex_end_done = re.compile(r"Epoch\s+(\d+)\s+done\.", re.IGNORECASE)
    rex_end_query = re.compile(
        r"Epoch\s+(\d+),\s+Queried expanded anytime results", re.IGNORECASE
    )

    for line in lines:
        ts = parse_bracket_timestamp(line, year_hint)
        if ts is None:
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue

        m_start = rex_start.search(line)
        if m_start:
            ep = int(m_start.group(1))
            if ep not in epoch_starts:
                epoch_starts[ep] = ts
            continue

        m_done = rex_end_done.search(line)
        if m_done:
            ep = int(m_done.group(1))
            epoch_ends[ep] = ts
            continue

        m_q = rex_end_query.search(line)
        if m_q:
            ep = int(m_q.group(1))
            if ep not in epoch_ends:
                epoch_ends[ep] = ts
            continue

    return epoch_starts, epoch_ends


# New: derive epoch runtimes from available stamps (fallback to deltas of 'done' stamps).
def _derive_epoch_runtimes_from_stamps(
    epoch_starts: Dict[int, datetime],
    epoch_ends: Dict[int, datetime],
) -> Dict[int, Optional[float]]:
    runtimes: Dict[int, Optional[float]] = {}
    ids = sorted(set(epoch_starts.keys()) | set(epoch_ends.keys()))
    prev_end: Optional[datetime] = None
    for ep in ids:
        st = epoch_starts.get(ep)
        en = epoch_ends.get(ep)
        dur: Optional[float] = None
        if st is not None and en is not None and en >= st:
            dur = (en - st).total_seconds()
        elif en is not None and prev_end is not None:
            # no explicit start; approximate by difference of consecutive 'done' timestamps
            dur = (en - prev_end).total_seconds()
        runtimes[ep] = dur
        if en is not None:
            prev_end = en
    return runtimes


def parse_queried_anytime_in_window(
    text: str,
    year_hint: int,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Dict[int, Dict[str, float]]:
    """
    Parse lines:
    [..]: Epoch E, Queried expanded anytime results: Test Acc: t, Val Acc: v, Train Acc: tr, Scaled Train Time: st
    Returns: {epoch: {"test": t, "val": v, "train": tr, "scaled_time": st}}
    """
    results: Dict[int, Dict[str, float]] = {}
    pat = re.compile(
        r"Epoch\s+(\d+),\s+Queried expanded anytime results:\s*Test Acc:\s*([0-9.+-Ee]+)\s*,\s*Val Acc:\s*([0-9.+-Ee]+)\s*,\s*Train Acc:\s*([0-9.+-Ee]+)\s*,\s*Scaled Train Time:\s*([0-9.+-Ee]+)",
        re.IGNORECASE,
    )
    for m in pat.finditer(text):
        ep = int(m.group(1))
        # bracket timestamp for window
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line = text[
            line_start if line_start >= 0 else 0 : line_end
            if line_end != -1
            else m.end()
        ]
        ts = parse_bracket_timestamp(line, year_hint)
        if ts is None:
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        results[ep] = {
            "test": float(m.group(2)),
            "val": float(m.group(3)),
            "train": float(m.group(4)),
            "scaled_time": float(m.group(5)),
        }
    return results


def parse_epoch_done_metrics_in_window(
    text: str,
    year_hint: int,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Dict[int, Dict[str, float]]:
    """
    Parse lines:
    [..]: Epoch E done. Train accuracy: X, Validation accuracy: Y
    Returns: {epoch: {"train_acc": X, "valid_acc": Y}}
    """
    results: Dict[int, Dict[str, float]] = {}
    pat = re.compile(
        r"Epoch\s+(\d+)\s+done\.\s*Train\s+accuracy\s*:\s*([0-9.+-Ee]+)\s*,\s*(?:Val(?:idation)?)\s+accuracy\s*:\s*([0-9.+-Ee]+)",
        re.IGNORECASE,
    )
    for m in pat.finditer(text):
        ep = int(m.group(1))
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end = text.find("\n", m.end())
        line = text[
            line_start if line_start >= 0 else 0 : line_end
            if line_end != -1
            else m.end()
        ]
        ts = parse_bracket_timestamp(line, year_hint)
        if ts is None:
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        results[ep] = {"train_acc": float(m.group(2)), "valid_acc": float(m.group(3))}
    return results


def load_errors_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def write_comprehensive_json(base_dir: str, merged: Dict[str, Any]) -> None:
    out_path = os.path.join(base_dir, "comprehensive_run_information.json")
    try:
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"Wrote {out_path}")
    except Exception as e:
        print(f"[WARN] Failed to write {out_path}: {e}")


# ------------------------------------------------------------
# Final-run parsing
# ------------------------------------------------------------


def find_final_save_dir(text: str) -> Optional[str]:
    """
    Prefer 'save path:  <path>' line; else parse
    'Saving architectural weight tensors: <path>/arch_weights.pt' and strip filename.
    """
    m = re.search(r"save path\s*:\s*([^\n\r]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"Saving architectural weight tensors:\s*([^\s]+)", text)
    if m2:
        p = m2.group(1).strip()
        return os.path.dirname(p)
    return None


def find_final_run_window(
    text: str,
) -> Tuple[Optional[datetime], Optional[datetime], int]:
    """
    Return (start_ts, end_ts, year_hint).
    - start: first '[MM/DD HH:MM:SS naslib]: Configuration is' line timestamp
    - end: last 'Training finished' or last epoch-done/queried line
    """
    year = extract_year_from_any_i_line(text) or datetime.now().year

    start_ts = None
    end_ts = None

    # Find configuration line (start anchor)
    for m in re.finditer(
        r"\[\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+naslib\]:\s+Configuration is", text
    ):
        start_ts = parse_bracket_timestamp(m.group(0), year)
        if start_ts:
            break

    # End anchors: prefer explicit 'Training finished'; fallback to last epoch end marker
    last_end_ts = None
    for m in re.finditer(
        r"\[\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+nl\.defaults.*\]:\s+Training finished",
        text,
    ):
        last_end_ts = parse_bracket_timestamp(m.group(0), year)
    if last_end_ts:
        end_ts = last_end_ts
    else:
        # Try last 'Epoch E done.' or 'Epoch E, Queried expanded anytime results'
        for m in re.finditer(
            r"\[\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}.*\]:\s+Epoch\s+\d+\s+done\.",
            text,
            re.IGNORECASE,
        ):
            last_end_ts = parse_bracket_timestamp(m.group(0), year)
        if not last_end_ts:
            for m in re.finditer(
                r"\[\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}.*\]:\s+Epoch\s+\d+,\s+Queried expanded anytime results",
                text,
                re.IGNORECASE,
            ):
                last_end_ts = parse_bracket_timestamp(m.group(0), year)
        end_ts = last_end_ts

    return start_ts, end_ts, year


# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------


def length_from_errors_arrays(err: Dict[str, Any]) -> int:
    candidates = []
    for key in (
        "train_acc",
        "valid_acc",
        "queried_val_acc",
        "queried_train_acc",
        "scaled_queried_train_time",
    ):
        v = err.get(key)
        if isinstance(v, list):
            candidates.append(len(v))
    return max(candidates) if candidates else 0


def _as_list(v: Any) -> List[float]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        return [float(v)]
    except Exception:
        return []


def _compare_numeric(a: float, b: float, tol: float = 1e-3) -> Tuple[bool, str]:
    if a is None or b is None:
        return False, "none"
    if abs(a - b) <= tol:
        return True, "direct"
    # try percent vs fraction
    if max(abs(a), abs(b)) > 1.0:
        if abs(a - b * 100.0) <= max(tol, 1e-2):
            return True, "b*100"
        if abs(a * 100.0 - b) <= max(tol, 1e-2):
            return True, "a*100"
    return False, "mismatch"


# New: absolute-tolerance comparator (for runtimes)
def _compare_series_abs_tol(
    name: str, err_vals: List[float], slurm_vals: List[float], abs_tol: float
) -> Dict[str, Any]:
    n = min(len(err_vals), len(slurm_vals))
    mismatches = []
    ok_cnt = 0
    for i in range(n):
        a, b = err_vals[i], slurm_vals[i]
        if a is None or b is None:
            mismatches.append({"index": i, "err": a, "slurm": b, "how": "none"})
            continue
        if abs(float(a) - float(b)) <= abs_tol:
            ok_cnt += 1
        else:
            mismatches.append(
                {"index": i, "err": a, "slurm": b, "how": f"|a-b|>{abs_tol}s"}
            )
    return {
        "series": name,
        "compared": n,
        "err_len": len(err_vals),
        "slurm_len": len(slurm_vals),
        "mismatches": mismatches,
        "match_count": ok_cnt,
        "abs_tol_sec": abs_tol,
    }


def _compare_series(
    name: str, err_vals: List[float], slurm_vals: List[float], tol: float = 1e-3
) -> Dict[str, Any]:
    n = min(len(err_vals), len(slurm_vals))
    mismatches = []
    used = {"direct": 0, "b*100": 0, "a*100": 0, "none": 0, "mismatch": 0}
    for i in range(n):
        ok, how = _compare_numeric(err_vals[i], slurm_vals[i], tol)
        used[how] = used.get(how, 0) + 1
        if not ok:
            mismatches.append(
                {"index": i, "err": err_vals[i], "slurm": slurm_vals[i], "how": how}
            )
    return {
        "series": name,
        "compared": n,
        "err_len": len(err_vals),
        "slurm_len": len(slurm_vals),
        "mismatches": mismatches,
        "match_count": n - len(mismatches),
        "strategy_counts": used,
    }


def build_sanity(
    epoch_starts,
    epoch_ends,
    err: Optional[Dict[str, Any]],
    *,
    slurm_queried: Optional[Dict[int, Dict[str, float]]] = None,
    slurm_done: Optional[Dict[int, Dict[str, float]]] = None,
    slurm_runtime: Optional[List[Optional[float]]] = None,
    runtime_tol_sec: float = 4.0,
    final_run: bool = False,
) -> Dict[str, Any]:
    """
    epoch_starts/epoch_ends can be:
      - Dict[int, datetime] (preferred), or
      - Iterable[int] of epoch ids.
    """

    # Normalize parsed epoch ids
    def _to_ids(x) -> List[int]:
        if x is None:
            return []
        if isinstance(x, dict):
            return list(x.keys())
        try:
            return list(x)
        except Exception:
            return []

    parsed_ids = sorted(set(_to_ids(epoch_starts)) | set(_to_ids(epoch_ends)))
    parsed_len = len(parsed_ids)
    err = err or {}
    err_len = length_from_errors_arrays(err)
    notes = []
    match = (err_len == parsed_len) if err_len and parsed_len else None
    if match is False:
        notes.append(
            f"Mismatch: errors.json epochs={err_len}, slurm parsed epochs={parsed_len}"
        )

    comparisons: List[Dict[str, Any]] = []

    # Compare queried metrics arrays if available
    if slurm_queried:
        q_epochs = sorted(slurm_queried.keys())
        s_test = [slurm_queried[e]["test"] for e in q_epochs]
        s_val = [slurm_queried[e]["val"] for e in q_epochs]
        s_train = [slurm_queried[e]["train"] for e in q_epochs]
        s_time = [slurm_queried[e]["scaled_time"] for e in q_epochs]
        comparisons.append(
            _compare_series(
                "queried_test_acc", _as_list(err.get("queried_test_acc")), s_test
            )
        )
        comparisons.append(
            _compare_series(
                "queried_val_acc", _as_list(err.get("queried_val_acc")), s_val
            )
        )
        comparisons.append(
            _compare_series(
                "queried_train_acc", _as_list(err.get("queried_train_acc")), s_train
            )
        )
        comparisons.append(
            _compare_series(
                "scaled_queried_train_time",
                _as_list(err.get("scaled_queried_train_time")),
                s_time,
            )
        )

    # For final runs: compare epoch-done train/valid accuracy
    if final_run and slurm_done:
        d_epochs = sorted(slurm_done.keys())
        s_train_acc = [slurm_done[e]["train_acc"] for e in d_epochs]
        s_valid_acc = [slurm_done[e]["valid_acc"] for e in d_epochs]
        comparisons.append(
            _compare_series("train_acc", _as_list(err.get("train_acc")), s_train_acc)
        )
        comparisons.append(
            _compare_series("valid_acc", _as_list(err.get("valid_acc")), s_valid_acc)
        )

    # Runtime comparison (absolute tolerance), for HPO and final runs
    runtime_check: Dict[str, Any] = {}
    if slurm_runtime:
        err_rt = _as_list(err.get("runtime"))
        # per-epoch abs tol
        runtime_check["per_epoch"] = _compare_series_abs_tol(
            "runtime", err_rt, slurm_runtime, runtime_tol_sec
        )
        # cumulative with abs tol on final value
        try:
            err_cum = [sum(err_rt[: i + 1]) for i in range(len(err_rt))]
            slurm_cum = [
                sum((x or 0.0) for x in slurm_runtime[: i + 1])
                for i in range(len(slurm_runtime))
            ]
            if err_cum and slurm_cum:
                ok = abs(err_cum[-1] - slurm_cum[-1]) <= runtime_tol_sec
                runtime_check["cumsum"] = {
                    "err_final": err_cum[-1],
                    "slurm_final": slurm_cum[-1],
                    "abs_tol_sec": runtime_tol_sec,
                    "match": bool(ok),
                }
        except Exception:
            pass

    return {
        "errors_json_epochs": err_len,
        "slurm_parsed_epochs": parsed_len,
        "match": match,
        "parsed_epoch_ids": parsed_ids,
        "notes": notes,
        "comparisons": comparisons,
        "runtime_check": runtime_check,
    }


# NEW: helpers to build final target dir under --final-out-root
def _tail_after_anchor(path: str, anchor: str) -> Optional[str]:
    """Return subpath after the given anchor directory name."""
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    if anchor in parts:
        idx = parts.index(anchor)
        tail_parts = parts[idx + 1 :]
        if tail_parts:
            return os.path.join(*tail_parts)
    return None


def _parse_final_filename(fname: str) -> Dict[str, Optional[str]]:
    """
    Expect filename like:
      final_hp_<optimizer>_<dataset>_<seed>_[<zc_proxy>]_<slurmId>.out
    Returns: optimizer, dataset, seed(str), zc_proxy, slurm_id(str)
    """
    base = os.path.basename(fname)
    if not (base.startswith("final_hp_") and base.endswith(".out")):
        return {
            "optimizer": None,
            "dataset": None,
            "seed": None,
            "zc_proxy": None,
            "slurm_id": None,
        }
    core = base[len("final_hp_") : -len(".out")]
    tokens = core.split("_")
    # find first numeric -> seed
    seed_idx = None
    for i, t in enumerate(tokens):
        if t.isdigit():
            seed_idx = i
            break
    if seed_idx is None or seed_idx < 1:
        return {
            "optimizer": None,
            "dataset": None,
            "seed": None,
            "zc_proxy": None,
            "slurm_id": None,
        }
    dataset = tokens[seed_idx - 1]
    optimizer = "_".join(tokens[: seed_idx - 1]) if seed_idx - 1 > 0 else None
    seed = tokens[seed_idx]
    zc_proxy = None
    slurm_id = None
    tail = tokens[seed_idx + 1 :]
    for t in tail:
        if t.isdigit():
            if slurm_id is None:
                slurm_id = t
        else:
            if zc_proxy is None:
                zc_proxy = t
    return {
        "optimizer": optimizer,
        "dataset": dataset,
        "seed": seed,
        "zc_proxy": zc_proxy,
        "slurm_id": slurm_id,
    }


def _parse_search_space_from_text(text: str) -> Optional[str]:
    m = re.search(r"\bsearch_space\s*:\s*([^\s]+)", text)
    return m.group(1).strip() if m else None


def _derive_final_target_dir(
    text: str, final_out_root: str, slurm_path: str
) -> Optional[str]:
    """
    Build target directory under final_out_root by:
      1) Using 'save path:' and replacing the 'result_final_hp' root with final_out_root, or
      2) Reconstructing from filename + 'search_space:' in log.
    """
    save_dir = find_final_save_dir(text)
    if save_dir:
        tail = _tail_after_anchor(save_dir, "result_final_hp")
        if tail:
            return os.path.join(final_out_root, tail)
        # If no anchor, try to recover last 5 path components: <optimizer>/<zc>/<search_space>/<dataset>/<seed>
        comps = os.path.normpath(save_dir).split(os.sep)
        if len(comps) >= 5:
            return os.path.join(final_out_root, *comps[-5:])
    # Fallback: from filename + search_space
    meta = _parse_final_filename(slurm_path)
    search_space = _parse_search_space_from_text(text)
    if all(
        [meta.get("optimizer"), meta.get("dataset"), meta.get("seed"), search_space]
    ):
        zc = meta.get("zc_proxy") or "params"
        return os.path.join(
            final_out_root,
            meta["optimizer"],
            zc,
            search_space,
            meta["dataset"],
            meta["seed"],
        )
    return None


def _get_chunk_id_from_filename(path: str) -> int:
    """
    The trailing numeric token before '.out' is treated as chunk id.
    final_hp_*_<chunk>.out
    """
    base = os.path.basename(path)
    m = re.search(r"_([0-9]+)\.out$", base)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _merge_epochs_preferring_higher_chunk(
    existing: Dict[int, Dict[str, Any]],
    incoming: Dict[int, Dict[str, Any]],
    existing_chunk: int,
    incoming_chunk: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Merge two epoch->data maps; when both have the same epoch, keep data from the higher chunk id.
    Data dicts are arbitrary (e.g., {"start": iso, "end": iso, "dur": sec} or queried values).
    """
    out: Dict[int, Dict[str, Any]] = {}
    # start with existing
    for ep, data in existing.items():
        out[ep] = dict(data)
        out[ep]["_chunk"] = existing_chunk
    # overlay incoming
    for ep, data in incoming.items():
        prev = out.get(ep)
        if (prev is None) or (prev.get("_chunk", -1) <= incoming_chunk):
            d = dict(data)
            d["_chunk"] = incoming_chunk
            out[ep] = d
    # strip helper
    for ep in list(out.keys()):
        out[ep].pop("_chunk", None)
    return out


def _arrays_to_epoch_map(ids: List[int], **arrays) -> Dict[int, Dict[str, Any]]:
    emap: Dict[int, Dict[str, Any]] = {}
    for i, ep in enumerate(ids):
        emap[ep] = {k: (v[i] if i < len(v) else None) for k, v in arrays.items()}
    return emap


def _epoch_map_to_arrays(
    emap: Dict[int, Dict[str, Any]],
) -> Tuple[List[int], Dict[str, List[Any]]]:
    ids = sorted(emap.keys())
    keys = set()
    for d in emap.values():
        keys.update(d.keys())
    arrays: Dict[str, List[Any]] = {k: [] for k in sorted(keys)}
    for ep in ids:
        for k in arrays.keys():
            arrays[k].append(emap[ep].get(k))
    return ids, arrays


# ------------------------------------------------------------
# Final-run parsing
# ------------------------------------------------------------


def find_final_save_dir(text: str) -> Optional[str]:
    """
    Prefer 'save path:  <path>' line; else parse
    'Saving architectural weight tensors: <path>/arch_weights.pt' and strip filename.
    """
    m = re.search(r"save path\s*:\s*([^\n\r]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"Saving architectural weight tensors:\s*([^\s]+)", text)
    if m2:
        p = m2.group(1).strip()
        return os.path.dirname(p)
    return None


# NEW: helpers to build final target dir under --final-out-root
def _tail_after_anchor(path: str, anchor: str) -> Optional[str]:
    """Return subpath after the given anchor directory name."""
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)
    if anchor in parts:
        idx = parts.index(anchor)
        tail_parts = parts[idx + 1 :]
        if tail_parts:
            return os.path.join(*tail_parts)
    return None


def _parse_final_filename(fname: str) -> Dict[str, Optional[str]]:
    """
    Expect filename like:
      final_hp_<optimizer>_<dataset>_<seed>_[<zc_proxy>]_<slurmId>.out
    Returns: optimizer, dataset, seed(str), zc_proxy, slurm_id(str)
    """
    base = os.path.basename(fname)
    if not (base.startswith("final_hp_") and base.endswith(".out")):
        return {
            "optimizer": None,
            "dataset": None,
            "seed": None,
            "zc_proxy": None,
            "slurm_id": None,
        }
    core = base[len("final_hp_") : -len(".out")]
    tokens = core.split("_")
    # find first numeric -> seed
    seed_idx = None
    for i, t in enumerate(tokens):
        if t.isdigit():
            seed_idx = i
            break
    if seed_idx is None or seed_idx < 1:
        return {
            "optimizer": None,
            "dataset": None,
            "seed": None,
            "zc_proxy": None,
            "slurm_id": None,
        }
    dataset = tokens[seed_idx - 1]
    optimizer = "_".join(tokens[: seed_idx - 1]) if seed_idx - 1 > 0 else None
    seed = tokens[seed_idx]
    zc_proxy = None
    slurm_id = None
    tail = tokens[seed_idx + 1 :]
    for t in tail:
        if t.isdigit():
            if slurm_id is None:
                slurm_id = t
        else:
            if zc_proxy is None:
                zc_proxy = t
    return {
        "optimizer": optimizer,
        "dataset": dataset,
        "seed": seed,
        "zc_proxy": zc_proxy,
        "slurm_id": slurm_id,
    }


def _parse_search_space_from_text(text: str) -> Optional[str]:
    m = re.search(r"\bsearch_space\s*:\s*([^\s]+)", text)
    return m.group(1).strip() if m else None


def _derive_final_target_dir(
    text: str, final_out_root: str, slurm_path: str
) -> Optional[str]:
    """
    Build target directory under final_out_root by:
      1) Using 'save path:' and replacing the 'result_final_hp' root with final_out_root, or
      2) Reconstructing from filename + 'search_space:' in log.
    """
    save_dir = find_final_save_dir(text)
    if save_dir:
        tail = _tail_after_anchor(save_dir, "result_final_hp")
        if tail:
            return os.path.join(final_out_root, tail)
        # If no anchor, try to recover last 5 path components: <optimizer>/<zc>/<search_space>/<dataset>/<seed>
        comps = os.path.normpath(save_dir).split(os.sep)
        if len(comps) >= 5:
            return os.path.join(final_out_root, *comps[-5:])
    # Fallback: from filename + search_space
    meta = _parse_final_filename(slurm_path)
    search_space = _parse_search_space_from_text(text)
    if all(
        [meta.get("optimizer"), meta.get("dataset"), meta.get("seed"), search_space]
    ):
        zc = meta.get("zc_proxy") or "params"
        return os.path.join(
            final_out_root,
            meta["optimizer"],
            zc,
            search_space,
            meta["dataset"],
            meta["seed"],
        )
    return None


def _get_chunk_id_from_filename(path: str) -> int:
    """
    The trailing numeric token before '.out' is treated as chunk id.
    final_hp_*_<chunk>.out
    """
    base = os.path.basename(path)
    m = re.search(r"_([0-9]+)\.out$", base)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def _merge_epochs_preferring_higher_chunk(
    existing: Dict[int, Dict[str, Any]],
    incoming: Dict[int, Dict[str, Any]],
    existing_chunk: int,
    incoming_chunk: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Merge two epoch->data maps; when both have the same epoch, keep data from the higher chunk id.
    Data dicts are arbitrary (e.g., {"start": iso, "end": iso, "dur": sec} or queried values).
    """
    out: Dict[int, Dict[str, Any]] = {}
    # start with existing
    for ep, data in existing.items():
        out[ep] = dict(data)
        out[ep]["_chunk"] = existing_chunk
    # overlay incoming
    for ep, data in incoming.items():
        prev = out.get(ep)
        if (prev is None) or (prev.get("_chunk", -1) <= incoming_chunk):
            d = dict(data)
            d["_chunk"] = incoming_chunk
            out[ep] = d
    # strip helper
    for ep in list(out.keys()):
        out[ep].pop("_chunk", None)
    return out


def _arrays_to_epoch_map(ids: List[int], **arrays) -> Dict[int, Dict[str, Any]]:
    emap: Dict[int, Dict[str, Any]] = {}
    for i, ep in enumerate(ids):
        emap[ep] = {k: (v[i] if i < len(v) else None) for k, v in arrays.items()}
    return emap


def _epoch_map_to_arrays(
    emap: Dict[int, Dict[str, Any]],
) -> Tuple[List[int], Dict[str, List[Any]]]:
    ids = sorted(emap.keys())
    keys = set()
    for d in emap.values():
        keys.update(d.keys())
    arrays: Dict[str, List[Any]] = {k: [] for k in sorted(keys)}
    for ep in ids:
        for k in arrays.keys():
            arrays[k].append(emap[ep].get(k))
    return ids, arrays


# ------------------------------------------------------------
# One-file processing
# ------------------------------------------------------------


def process_one_slurm_file(path: str, args: argparse.Namespace) -> None:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    is_hpo = detect_is_hpo(text)
    year_hint = extract_year_from_any_i_line(text) or datetime.now().year

    if is_hpo:
        study_name = extract_study_name(text)
        optimizer = search_space = dataset = None
        seed = None
        zcp_method = None
        if study_name:
            try:
                optimizer, search_space, dataset, seed, zcp_method = (
                    parse_study_identity(study_name)
                )
            except Exception as e:
                print(
                    f"[WARN] ({os.path.basename(path)}) Failed to parse study identity: {e}"
                )

        trials = find_hpo_trial_events(text)
        for tnum, t in sorted(trials.items(), key=lambda kv: kv[0]):
            epoch_starts, epoch_ends = parse_epoch_timestamps_in_window(
                text, year_hint, t.start, t.end
            )
            t.epoch_starts = epoch_starts
            t.epoch_ends = epoch_ends
            for ep in sorted(set(epoch_starts.keys()) & set(epoch_ends.keys())):
                t.epoch_durations[ep] = (
                    epoch_ends[ep] - epoch_starts[ep]
                ).total_seconds()
            # Fill missing using deltas of 'done' stamps
            dur_fb = _derive_epoch_runtimes_from_stamps(epoch_starts, epoch_ends)
            for ep, dv in dur_fb.items():
                if ep not in t.epoch_durations or t.epoch_durations[ep] is None:
                    if isinstance(dv, (int, float)):
                        t.epoch_durations[ep] = dv

            # Slurm metrics parsing in window
            slurm_queried = parse_queried_anytime_in_window(
                text, year_hint, t.start, t.end
            )
            slurm_done = parse_epoch_done_metrics_in_window(
                text, year_hint, t.start, t.end
            )

            # Determine trial dir and errors.json
            trial_dir = t.dirpath
            if not trial_dir and all(
                [optimizer, search_space, dataset, seed is not None]
            ):
                trial_dir = reconstruct_hpo_trial_dir(
                    args.hpo_out_root,
                    optimizer,
                    search_space,
                    dataset,
                    seed,
                    zcp_method,
                    tnum,
                )
            if not trial_dir:
                print(
                    f"[WARN] ({os.path.basename(path)}) Could not determine trial dir for trial {tnum}. Skipping."
                )
                continue

            errors_path = os.path.join(trial_dir, "errors.json")
            errors = load_errors_json(errors_path) or {}

            merged: Dict[str, Any] = dict(errors)
            merged.update(
                {
                    "slurm_source_file": os.path.abspath(path),
                    "slurm_trial_start": t.start.isoformat() if t.start else None,
                    "slurm_trial_end": t.end.isoformat() if t.end else None,
                    "slurm_trial_duration_seconds": (t.end - t.start).total_seconds()
                    if (t.start and t.end)
                    else None,
                    "trial_number": tnum,
                    "trial_budget": t.budget,
                    "trial_state": t.state,
                    "study_name": study_name,
                }
            )

            # Per-epoch wall-clock arrays
            if t.epoch_starts or t.epoch_ends:
                epochs_sorted = sorted(
                    set(list(t.epoch_starts.keys()) + list(t.epoch_ends.keys()))
                )
                merged["slurm_epoch_wallclock_start"] = [
                    t.epoch_starts.get(ep).isoformat()
                    if t.epoch_starts.get(ep)
                    else None
                    for ep in epochs_sorted
                ]
                merged["slurm_epoch_wallclock_end"] = [
                    t.epoch_ends.get(ep).isoformat() if t.epoch_ends.get(ep) else None
                    for ep in epochs_sorted
                ]
                merged["slurm_epoch_durations_seconds"] = [
                    t.epoch_durations.get(ep) for ep in epochs_sorted
                ]
                merged["slurm_epoch_ids"] = epochs_sorted

            # Attach slurm queried metrics
            if slurm_queried:
                q_epochs = sorted(slurm_queried.keys())
                merged["slurm_queried_epoch_ids"] = q_epochs
                merged["slurm_queried_test_acc"] = [
                    slurm_queried[e]["test"] for e in q_epochs
                ]
                merged["slurm_queried_val_acc"] = [
                    slurm_queried[e]["val"] for e in q_epochs
                ]
                merged["slurm_queried_train_acc"] = [
                    slurm_queried[e]["train"] for e in q_epochs
                ]
                merged["slurm_scaled_queried_train_time"] = [
                    slurm_queried[e]["scaled_time"] for e in q_epochs
                ]

            # Attach 'epoch done' metrics if present
            if slurm_done:
                d_epochs = sorted(slurm_done.keys())
                merged["slurm_epoch_done_ids"] = d_epochs
                merged["slurm_epoch_done_train_acc"] = [
                    slurm_done[e]["train_acc"] for e in d_epochs
                ]
                merged["slurm_epoch_done_valid_acc"] = [
                    slurm_done[e]["valid_acc"] for e in d_epochs
                ]

            # Build slurm_runtime list for sanity from durations
            slurm_runtime_list = merged.get("slurm_epoch_durations_seconds", [])

            merged["sanity"] = build_sanity(
                t.epoch_starts,
                t.epoch_ends,
                errors,
                slurm_queried=slurm_queried,
                slurm_done=None,  # HPO often lacks reliable 'done' metrics
                slurm_runtime=slurm_runtime_list,
                runtime_tol_sec=4.0,
                final_run=False,
            )

            if args.dry_run:
                print(
                    f"[DRY] Would write comprehensive_run_information.json next to: {errors_path}"
                )
            else:
                out_dir = os.path.dirname(errors_path)
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    pass
                write_comprehensive_json(out_dir, merged)

    else:
        # FINAL RUN
        # Decide target dir under --final-out-root
        target_dir = _derive_final_target_dir(text, args.final_out_root, path)
        if not target_dir:
            print(
                f"[WARN] ({os.path.basename(path)}) Could not determine final run target directory. Skipping."
            )
            return
        os.makedirs(target_dir, exist_ok=True)
        errors_path_candidates = [
            os.path.join(target_dir, "search", "errors.json"),
            os.path.join(target_dir, "errors.json"),
        ]
        errors_path = next(
            (p for p in errors_path_candidates if os.path.exists(p)),
            errors_path_candidates[-1],
        )

        errors = load_errors_json(errors_path) or {}

        run_start, run_end, year_hint = find_final_run_window(text)

        epoch_starts, epoch_ends = parse_epoch_timestamps_in_window(
            text, year_hint, run_start, run_end
        )
        slurm_queried = parse_queried_anytime_in_window(
            text, year_hint, run_start, run_end
        )
        slurm_done = parse_epoch_done_metrics_in_window(
            text, year_hint, run_start, run_end
        )

        epoch_durations: Dict[int, float] = {}
        for ep in sorted(set(epoch_starts.keys()) & set(epoch_ends.keys())):
            epoch_durations[ep] = (epoch_ends[ep] - epoch_starts[ep]).total_seconds()

        merged: Dict[str, Any] = dict(errors)
        merged.update(
            {
                "final_run": True,
                "slurm_source_file": os.path.abspath(path),
                "slurm_run_start": run_start.isoformat() if run_start else None,
                "slurm_run_end": run_end.isoformat() if run_end else None,
                "slurm_run_duration_seconds": (run_end - run_start).total_seconds()
                if (run_start and run_end)
                else None,
                "slurm_chunk_id": _get_chunk_id_from_filename(path),
            }
        )

        if epoch_starts or epoch_ends:
            epochs_sorted = sorted(
                set(list(epoch_starts.keys()) + list(epoch_ends.keys()))
            )
            merged["slurm_epoch_wallclock_start"] = [
                epoch_starts.get(ep).isoformat() if epoch_starts.get(ep) else None
                for ep in epochs_sorted
            ]
            merged["slurm_epoch_wallclock_end"] = [
                epoch_ends.get(ep).isoformat() if epoch_ends.get(ep) else None
                for ep in epochs_sorted
            ]
            # Derive/fill durations
            if not epoch_durations:
                epoch_durations = {}
            # direct if both stamps exist
            for ep in epochs_sorted:
                st = epoch_starts.get(ep)
                en = epoch_ends.get(ep)
                if st and en and en >= st:
                    epoch_durations[ep] = (en - st).total_seconds()
            # fill missing via deltas of 'done' stamps
            dur_fb = _derive_epoch_runtimes_from_stamps(epoch_starts, epoch_ends)
            for ep, dv in dur_fb.items():
                if ep not in epoch_durations or epoch_durations[ep] is None:
                    if isinstance(dv, (int, float)):
                        epoch_durations[ep] = dv
            merged["slurm_epoch_durations_seconds"] = [
                epoch_durations.get(ep) for ep in epochs_sorted
            ]
            merged["slurm_epoch_ids"] = epochs_sorted

        if slurm_queried:
            q_epochs = sorted(slurm_queried.keys())
            merged["slurm_queried_epoch_ids"] = q_epochs
            merged["slurm_queried_test_acc"] = [
                slurm_queried[e]["test"] for e in q_epochs
            ]
            merged["slurm_queried_val_acc"] = [
                slurm_queried[e]["val"] for e in q_epochs
            ]
            merged["slurm_queried_train_acc"] = [
                slurm_queried[e]["train"] for e in q_epochs
            ]
            merged["slurm_scaled_queried_train_time"] = [
                slurm_queried[e]["scaled_time"] for e in q_epochs
            ]

        if slurm_done:
            d_epochs = sorted(slurm_done.keys())
            merged["slurm_epoch_done_ids"] = d_epochs
            merged["slurm_epoch_done_train_acc"] = [
                slurm_done[e]["train_acc"] for e in d_epochs
            ]
            merged["slurm_epoch_done_valid_acc"] = [
                slurm_done[e]["valid_acc"] for e in d_epochs
            ]

        # Merge with existing comprehensive JSON (multi-chunk support, prefer newer chunk on overlap)
        comp_path = os.path.join(target_dir, "comprehensive_run_information.json")
        prev = load_errors_json(comp_path) or {}
        if prev:
            # Merge wallclock per-epoch
            incoming_chunk = merged.get("slurm_chunk_id", 0)
            existing_chunk = prev.get("slurm_chunk_id", 0)
            # wallclock
            new_eids = merged.get("slurm_epoch_ids", [])
            new_wmap = _arrays_to_epoch_map(
                new_eids,
                start=merged.get("slurm_epoch_wallclock_start", []),
                end=merged.get("slurm_epoch_wallclock_end", []),
                dur=merged.get("slurm_epoch_durations_seconds", []),
            )
            old_eids = prev.get("slurm_epoch_ids", [])
            old_wmap = _arrays_to_epoch_map(
                old_eids,
                start=prev.get("slurm_epoch_wallclock_start", []),
                end=prev.get("slurm_epoch_wallclock_end", []),
                dur=prev.get("slurm_epoch_durations_seconds", []),
            )
            merged_wmap = _merge_epochs_preferring_higher_chunk(
                old_wmap, new_wmap, existing_chunk, incoming_chunk
            )
            eids, warrays = _epoch_map_to_arrays(merged_wmap)
            merged["slurm_epoch_ids"] = eids
            merged["slurm_epoch_wallclock_start"] = warrays.get("start", [])
            merged["slurm_epoch_wallclock_end"] = warrays.get("end", [])
            merged["slurm_epoch_durations_seconds"] = warrays.get("dur", [])

            # queried
            new_qids = merged.get("slurm_queried_epoch_ids", [])
            new_qmap = _arrays_to_epoch_map(
                new_qids,
                test=merged.get("slurm_queried_test_acc", []),
                val=merged.get("slurm_queried_val_acc", []),
                train=merged.get("slurm_queried_train_acc", []),
                scaled_time=merged.get("slurm_scaled_queried_train_time", []),
            )
            old_qids = prev.get("slurm_queried_epoch_ids", [])
            old_qmap = _arrays_to_epoch_map(
                old_qids,
                test=prev.get("slurm_queried_test_acc", []),
                val=prev.get("slurm_queried_val_acc", []),
                train=prev.get("slurm_queried_train_acc", []),
                scaled_time=prev.get("slurm_scaled_queried_train_time", []),
            )
            merged_qmap = _merge_epochs_preferring_higher_chunk(
                old_qmap, new_qmap, existing_chunk, incoming_chunk
            )
            qids, qarrays = _epoch_map_to_arrays(merged_qmap)
            if qids:
                merged["slurm_queried_epoch_ids"] = qids
                merged["slurm_queried_test_acc"] = qarrays.get("test", [])
                merged["slurm_queried_val_acc"] = qarrays.get("val", [])
                merged["slurm_queried_train_acc"] = qarrays.get("train", [])
                merged["slurm_scaled_queried_train_time"] = qarrays.get(
                    "scaled_time", []
                )

            # epoch-done
            new_dids = merged.get("slurm_epoch_done_ids", [])
            new_dmap = _arrays_to_epoch_map(
                new_dids,
                train_acc=merged.get("slurm_epoch_done_train_acc", []),
                valid_acc=merged.get("slurm_epoch_done_valid_acc", []),
            )
            old_dids = prev.get("slurm_epoch_done_ids", [])
            old_dmap = _arrays_to_epoch_map(
                old_dids,
                train_acc=prev.get("slurm_epoch_done_train_acc", []),
                valid_acc=prev.get("slurm_epoch_done_valid_acc", []),
            )
            merged_dmap = _merge_epochs_preferring_higher_chunk(
                old_dmap, new_dmap, existing_chunk, incoming_chunk
            )
            dids, darrays = _epoch_map_to_arrays(merged_dmap)
            if dids:
                merged["slurm_epoch_done_ids"] = dids
                merged["slurm_epoch_done_train_acc"] = darrays.get("train_acc", [])
                merged["slurm_epoch_done_valid_acc"] = darrays.get("valid_acc", [])

            # Merge run window (min start, max end)
            def _parse_iso(x):
                try:
                    return datetime.fromisoformat(x) if x else None
                except Exception:
                    return None

            prev_start = _parse_iso(prev.get("slurm_run_start"))
            prev_end = _parse_iso(prev.get("slurm_run_end"))
            cur_start = _parse_iso(merged.get("slurm_run_start"))
            cur_end = _parse_iso(merged.get("slurm_run_end"))
            win_start = min([t for t in [prev_start, cur_start] if t], default=None)
            win_end = max([t for t in [prev_end, cur_end] if t], default=None)
            merged["slurm_run_start"] = win_start.isoformat() if win_start else None
            merged["slurm_run_end"] = win_end.isoformat() if win_end else None
            if win_start and win_end:
                merged["slurm_run_duration_seconds"] = (
                    win_end - win_start
                ).total_seconds()
            # Track chunks
            chunks = prev.get("slurm_merge", {}).get("chunks", [])
            chunks = list(chunks) if isinstance(chunks, list) else []
            chunks.append(
                {
                    "chunk_id": incoming_chunk,
                    "source_file": os.path.abspath(path),
                    "epoch_ids": merged.get("slurm_epoch_ids", []),
                    "run_start": merged.get("slurm_run_start"),
                    "run_end": merged.get("slurm_run_end"),
                }
            )
            merged["slurm_merge"] = {"chunks": chunks, "last_chunk_id": incoming_chunk}

        # Build merged slurm_queried and slurm_done dicts for sanity
        mq: Optional[Dict[int, Dict[str, float]]] = None
        md: Optional[Dict[int, Dict[str, float]]] = None

        qids = merged.get("slurm_queried_epoch_ids") or []
        qt = merged.get("slurm_queried_test_acc") or []
        qv = merged.get("slurm_queried_val_acc") or []
        qtr = merged.get("slurm_queried_train_acc") or []
        qst = merged.get("slurm_scaled_queried_train_time") or []
        if qids and (qt or qv or qtr or qst):
            mq = {}
            for i, ep in enumerate(qids):
                mq[int(ep)] = {
                    "test": qt[i] if i < len(qt) else None,
                    "val": qv[i] if i < len(qv) else None,
                    "train": qtr[i] if i < len(qtr) else None,
                    "scaled_time": qst[i] if i < len(qst) else None,
                }

        dids = merged.get("slurm_epoch_done_ids") or []
        dtr = merged.get("slurm_epoch_done_train_acc") or []
        dvl = merged.get("slurm_epoch_done_valid_acc") or []
        if dids and (dtr or dvl):
            md = {}
            for i, ep in enumerate(dids):
                md[int(ep)] = {
                    "train_acc": dtr[i] if i < len(dtr) else None,
                    "valid_acc": dvl[i] if i < len(dvl) else None,
                }

        # Slurm runtime list (durations) for sanity
        slurm_runtime_list = merged.get("slurm_epoch_durations_seconds", [])

        merged["sanity"] = build_sanity(
            merged.get("slurm_epoch_ids", []),
            merged.get("slurm_epoch_ids", []),
            errors,
            slurm_queried=mq,
            slurm_done=md,
            slurm_runtime=slurm_runtime_list,
            runtime_tol_sec=4.0,
            final_run=True,
        )

        if args.dry_run:
            print(
                f"[DRY] Would write comprehensive_run_information.json to: {os.path.join(target_dir, 'comprehensive_run_information.json')}"
            )
        else:
            write_comprehensive_json(target_dir, merged)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Create comprehensive_run_information.json next to errors.json using Slurm logs."
    )
    ap.add_argument("--slurm", help="Path to a single Slurm .out log file to parse.")
    ap.add_argument(
        "--slurm-dir",
        help="Directory containing Slurm .out files to parse recursively.",
    )
    ap.add_argument(
        "--pattern",
        default="*.out",
        help="Glob pattern when using --slurm-dir (default: *.out).",
    )
    ap.add_argument(
        "--hpo-out-root",
        default="naslib/optimizers/oneshot/gsparsity/results_wide_hpo_queried_val_acc",
        help="HPO results out root (for reconstructing trial dirs).",
    )
    ap.add_argument(
        "--final-out-root",
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp_queried_val_acc",
        help="Final results out root (not strictly needed if 'save path:' is in log).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print planned actions.",
    )
    args = ap.parse_args()

    files: List[str] = []
    if args.slurm:
        files.append(args.slurm)
    if args.slurm_dir:
        files.extend(
            glob(os.path.join(args.slurm_dir, "**", args.pattern), recursive=True)
        )

    if not files:
        print("Provide --slurm or --slurm-dir.")
        return

    for fp in sorted(set(files)):
        if not os.path.isfile(fp):
            continue
        try:
            process_one_slurm_file(fp, args)
        except Exception as e:
            print(f"[WARN] Failed to process {fp}: {e}")


if __name__ == "__main__":
    main()
