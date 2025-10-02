#!/usr/bin/env python3
import argparse
import ast
import os
import re
from glob import glob
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_filename(fname: str) -> Dict[str, Optional[str]]:
    """
    Expect filename like:
      final_hp_<name>_<dataset>_<seed>_[<zc_proxy>]_<slurmId>.out
    Examples:
      final_hp_zcp_gsparsity_cifar100_1544457859_jacov_1251794.out
      final_hp_zcp_gsparsity_ImageNet16-120_1544457859_jacov_1251800.out
    Returns fields: name, dataset, seed, zc_proxy (optional), slurm_id (optional)
    """
    base = os.path.basename(fname)
    if not base.startswith("final_hp_") or not base.endswith(".out"):
        return {
            "name": None,
            "dataset": None,
            "seed": None,
            "zc_proxy": None,
            "slurm_id": None,
        }

    core = base[len("final_hp_") : -len(".out")]
    tokens = core.split("_")
    # Find first numeric token => seed
    seed_idx = None
    for i, t in enumerate(tokens):
        if t.isdigit():
            seed_idx = i
            break

    if seed_idx is None or seed_idx < 1:
        return {
            "name": None,
            "dataset": None,
            "seed": None,
            "zc_proxy": None,
            "slurm_id": None,
        }

    dataset = tokens[seed_idx - 1]
    name = "_".join(tokens[: seed_idx - 1]) if seed_idx - 1 > 0 else None
    seed = tokens[seed_idx]

    # Parse tokens after seed
    zc_proxy = None
    slurm_id = None
    tail = tokens[seed_idx + 1 :]
    # Common ZC proxies; treat first non-numeric in tail as proxy, first numeric in tail as slurm id
    for t in tail:
        if t.isdigit():
            if slurm_id is None:
                slurm_id = t
        else:
            if zc_proxy is None:
                zc_proxy = t

    return {
        "name": name,
        "dataset": dataset,
        "seed": seed,
        "zc_proxy": zc_proxy,
        "slurm_id": slurm_id,
    }


def extract_list(pattern_name: str, text: str) -> Optional[List[float]]:
    """
    Extract a Python-like list from text after a label, e.g.
      'Train accuracies: [1.0, 2.0, ...]'
    Uses DOTALL to allow multi-line lists.
    """
    # Greedy bracket match up to the first closing bracket
    m = re.search(
        rf"{pattern_name}\s*:\s*(\[[^\]]*\])", text, re.IGNORECASE | re.DOTALL
    )
    if not m:
        return None
    try:
        lst = ast.literal_eval(m.group(1))
        if isinstance(lst, list):
            # Ensure floats
            return [float(x) for x in lst]
    except Exception:
        return None
    return None


def parse_wallclock_string(s: str) -> Optional[int]:
    """
    Parse wall-clock strings:
      - D-HH:MM:SS[.fff]
      - HHH:MM:SS[.fff]
      - M:SS[.fff]  (from '/usr/bin/time -v')
    """
    s = s.strip()
    # D-HH:MM:SS(.fff)
    if "-" in s:
        day_part, hms = s.split("-", 1)
        try:
            d = int(day_part)
            parts = hms.split(":")
            if len(parts) != 3:
                return None
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
            return d * 86400 + h * 3600 + m * 60 + int(round(sec))
        except Exception:
            return None
    # HHH:MM:SS(.fff) or M:SS(.fff)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
            return h * 3600 + m * 60 + int(round(sec))
        if len(parts) == 2:
            m = int(parts[0])
            sec = float(parts[1])
            return m * 60 + int(round(sec))
    except Exception:
        return None
    return None


def format_wallclock_seconds(total_seconds: Optional[int]) -> Optional[str]:
    if total_seconds is None:
        return None
    total = int(total_seconds)
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    if d > 0:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_wallclock(text: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Look for multiple wall-clock patterns; return (matched_string, seconds).
    Takes the last occurrence in the file.
    """
    patterns = [
        # Typical SLURM summary line
        r"Job\s+Wall[- ]clock\s+time:\s*(\d+-\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2}:\d{2})",
        # '/usr/bin/time -v'
        r"Elapsed\s*\(wall clock\)\s*time\s*\([^)]*\)\s*:\s*(\d+-\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2}:\d{2}|\d{1,3}:\d{2}(?:\.\d+)?)",
        # Generic "Elapsed" or "Elapsed time"
        r"Elapsed(?:\s*time)?\s*:?\s*(\d+-\d{1,2}:\d{2}:\d{2}|\d{1,4}:\d{2}:\d{2}|\d{1,3}:\d{2}(?:\.\d+)?)",
    ]
    last_match = None
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.MULTILINE):
            last_match = m.group(1).strip()
    if not last_match:
        return None, None
    secs = parse_wallclock_string(last_match)
    return last_match, secs


def extract_epoch_logs(
    text: str,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Fallback: parse lines like
      'Epoch 28 done. Train accuracy: 22.32249, Validation accuracy: 14.59651'
    Returns train/val lists indexed by epoch (1-based in logs -> list position-1).
    """
    # Accept 'done' or 'completed', optional punctuation, flexible spacing, case-insensitive.
    pat = re.compile(
        r"Epoch\s+(\d+)\s*(?:done|completed)?\.?\s*.*?"
        r"Train\s+accuracy\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"
        r"(?:Val(?:idation)?)\s+accuracy\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    )
    entries: Dict[int, Tuple[float, float]] = {}
    for m in pat.finditer(text):
        ep = int(m.group(1))
        tr = float(m.group(2))
        va = float(m.group(3))
        entries[ep] = (tr, va)  # keep last occurrence per epoch

    if not entries:
        return None, None

    max_ep = max(entries.keys())
    train = [None] * max_ep
    val = [None] * max_ep
    for ep, (tr, va) in entries.items():
        idx = ep - 1
        if idx >= 0:
            train[idx] = tr
            val[idx] = va

    # Compact None tails if needed
    while train and train[-1] is None and val and val[-1] is None:
        train.pop()
        val.pop()

    # Replace remaining None with previous known (or drop if none known)
    last_tr = last_va = None
    for i in range(len(train)):
        if train[i] is None:
            train[i] = last_tr if last_tr is not None else 0.0
        else:
            last_tr = train[i]
        if val[i] is None:
            val[i] = last_va if last_va is not None else 0.0
        else:
            last_va = val[i]

    return train, val


def parse_file(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Filename-derived fields
    meta = parse_filename(path)

    # Content-derived fields
    wallclock_str = None
    wallclock_seconds: Optional[int] = None
    arch_hash = None
    best_val_acc = None

    # Robust wall-clock extraction (supports D-HH:MM:SS and variants)
    wc_str, wc_secs = extract_wallclock(content)
    wallclock_str = wc_str
    wallclock_seconds = wc_secs

    m_hash = re.search(r"Final architecture hash:\s*(\([^)]+\))", content)
    if m_hash:
        arch_hash = m_hash.group(1)

    m_best = re.search(
        r"Best model validation accuracy:\s*([+-]?\d+(?:\.\d+)?)",
        content,
        re.IGNORECASE,
    )
    if m_best:
        try:
            best_val_acc = float(m_best.group(1))
        except ValueError:
            best_val_acc = None

    train_accs = extract_list("Train accuracies", content)
    val_accs = extract_list("Validation accuracies", content)

    # Fallback: per-epoch log lines if list form is missing
    if not (train_accs and val_accs):
        tr_fb, va_fb = extract_epoch_logs(content)
        if tr_fb and va_fb:
            train_accs, val_accs = tr_fb, va_fb

    epochs = max(len(train_accs) if train_accs else 0, len(val_accs) if val_accs else 0)

    row: Dict[str, object] = {
        "file_path": path,
        "file_name": os.path.basename(path),
        "name": meta.get("name"),
        "dataset": meta.get("dataset"),
        "seed": int(meta["seed"])
        if meta.get("seed") and meta["seed"].isdigit()
        else None,
        "zc_proxy": meta.get("zc_proxy"),
        "slurm_id": int(meta["slurm_id"])
        if meta.get("slurm_id") and meta["slurm_id"].isdigit()
        else None,
        "wallclock": wallclock_str,  # per-file string
        "wallclock_seconds": wallclock_seconds,  # per-file numeric for aggregation
        "final_arch_hash": arch_hash,
        "best_val_acc": best_val_acc,
        "epochs": epochs,
    }

    # Expand accuracies into one column per epoch
    if train_accs:
        for i, v in enumerate(train_accs, start=1):
            row[f"train_acc_ep{i}"] = v
    if val_accs:
        for i, v in enumerate(val_accs, start=1):
            row[f"val_acc_ep{i}"] = v

    return row


def find_files(input_dir: str, pattern: str, recursive: bool) -> List[str]:
    if recursive:
        return glob(os.path.join(input_dir, "**", pattern), recursive=True)
    return glob(os.path.join(input_dir, pattern))


def aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple slurm chunks for the same (name, dataset, seed, zc_proxy).
    - Sum wallclock_seconds across chunks.
    - Use the last chunk (highest slurm_id) for end-state values and accuracies.
    """
    if df.empty:
        return df

    group_cols = ["name", "dataset", "seed", "zc_proxy"]

    # Ensure slurm_id numeric for ordering; NaN -> -inf ordering
    df["slurm_id_numeric"] = pd.to_numeric(df["slurm_id"], errors="coerce")

    aggregated_rows = []
    for _, g in df.groupby(group_cols, dropna=False):
        g_sorted = g.sort_values(
            by=["slurm_id_numeric", "file_name"], na_position="first"
        )
        last = g_sorted.tail(1).iloc[0]

        # Sum wallclock seconds across chunks (treat missing as 0)
        sum_secs = (
            pd.to_numeric(g_sorted["wallclock_seconds"], errors="coerce")
            .fillna(0)
            .sum()
        )
        sum_secs = int(sum_secs)
        sum_wc_str = format_wallclock_seconds(sum_secs)

        # Build result from last row, then adjust fields
        result = last.to_dict()
        result["wallclock"] = sum_wc_str
        result["wallclock_seconds"] = sum_secs
        result["num_chunks"] = int(len(g_sorted))
        # First/last slurm ids for reference
        sid_min = pd.to_numeric(g_sorted["slurm_id"], errors="coerce").min()
        sid_max = pd.to_numeric(g_sorted["slurm_id"], errors="coerce").max()
        result["slurm_id_first"] = int(sid_min) if pd.notna(sid_min) else None
        result["slurm_id_last"] = int(sid_max) if pd.notna(sid_max) else None

        aggregated_rows.append(result)

    agg_df = pd.DataFrame(aggregated_rows)

    # Drop helper column
    if "slurm_id_numeric" in agg_df.columns:
        agg_df = agg_df.drop(columns=["slurm_id_numeric"])

    return agg_df


def main():
    ap = argparse.ArgumentParser(
        description="Parse slurm .out files and export an Excel table."
    )
    ap.add_argument(
        "--input_dir",
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp/slurm",
        help="Directory containing slurm out files (e.g., naslib/optimizers/oneshot/gsparsity/result_final_hp/slurm)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default="slurm_parsed_results.xlsx",
        help="Output Excel file path (default: slurm_parsed_results.xlsx)",
    )
    ap.add_argument(
        "-p",
        "--pattern",
        default="final_hp_*.out",
        help="Glob pattern for files to parse (default: final_hp_*.out)",
    )
    ap.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Search recursively for files matching the pattern",
    )
    args = ap.parse_args()

    files = find_files(args.input_dir, args.pattern, args.recursive)
    if not files:
        print(f"No files found in {args.input_dir} matching pattern '{args.pattern}'.")
        return

    rows = []
    for fp in sorted(files):
        try:
            rows.append(parse_file(fp))
        except Exception as e:
            print(f"Failed to parse '{fp}': {e}")

    if not rows:
        print("No rows parsed. Nothing to write.")
        return

    df = pd.DataFrame(rows)

    # Aggregate sequential chunks per (name, dataset, seed, zc_proxy)
    agg_df = aggregate_rows(df)

    # Sort helpful columns first
    preferred_order = [
        "file_name",
        "file_path",
        "name",
        "dataset",
        "seed",
        "zc_proxy",
        "num_chunks",
        "slurm_id_first",
        "slurm_id_last",
        "slurm_id",
        "wallclock",
        "final_arch_hash",
        "best_val_acc",
        "epochs",
    ]
    # Then dynamic epoch columns in order from the aggregated df
    train_cols = sorted(
        [c for c in agg_df.columns if c.startswith("train_acc_ep")],
        key=lambda x: int(x.split("ep")[-1]),
    )
    val_cols = sorted(
        [c for c in agg_df.columns if c.startswith("val_acc_ep")],
        key=lambda x: int(x.split("ep")[-1]),
    )

    # Keep wallclock_seconds for debugging but place it after wallclock
    if (
        "wallclock_seconds" in agg_df.columns
        and "wallclock_seconds" not in preferred_order
    ):
        preferred_order.insert(
            preferred_order.index("wallclock") + 1, "wallclock_seconds"
        )

    other_cols = [
        c
        for c in agg_df.columns
        if c not in set(preferred_order + train_cols + val_cols)
    ]

    # Only include columns that actually exist
    final_cols = (
        [c for c in preferred_order if c in agg_df.columns]
        + train_cols
        + val_cols
        + other_cols
    )
    agg_df = agg_df[final_cols]

    # Write Excel
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        agg_df.to_excel(writer, index=False, sheet_name="results")

    print(f"Wrote {len(agg_df)} aggregated rows (from {len(df)} files) to {out_path}")


if __name__ == "__main__":
    main()
