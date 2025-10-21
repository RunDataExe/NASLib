import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

# Extra time budget for Random Search (in seconds)
EXTRA_T_RANDOM_SEARCH = 48 * 3600  # 48 hours

def latex_escape(text: Optional[str]) -> str:
    if text is None:
        return ""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = str(text)
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def _normalize_accuracy_scale(acc: np.ndarray) -> np.ndarray:
    if acc.size == 0:
        return acc
    finite = acc[np.isfinite(acc)]
    if finite.size == 0:
        return acc
    return acc / 100.0 if np.nanmax(finite) > 1.5 else acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-time comparison for final runs. For each dataset, choose the maximum common budget T "
            "(min last cumulative time across all runs in that dataset; cumulative search + eval) and report: "
            "Best-within-T Train/Val (mean ± std) and AUC/T Train/Val (mean ± std)."
        )
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp",
        help="Root directory that contains run folders with errors.json files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp/fixed_time_performance_summary.xlsx",
        help="Output file path (.xlsx or .csv).",
    )
    parser.add_argument(
        "--fractional",
        action="store_true",
        help="Keep accuracies in [0,1]. By default accuracies are reported in percent.",
    )
    parser.add_argument(
        "--latex_output",
        type=str,
        default="",
        help="Optional path for the LaTeX table (.txt). Defaults to output base name.",
    )
    return parser.parse_args()


def find_error_files(root_dir: str) -> Tuple[Dict, ...]:
    files: List[Dict] = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "errors.json" not in filenames:
            continue
        full_path = os.path.join(dirpath, "errors.json")
        relative_path = os.path.relpath(full_path, root_dir)
        parts = relative_path.split(os.sep)
        # With zcp: {optimizer}/{zcp}/{ss}/{dataset}/{seed}/errors.json
        # Without:  {optimizer}/{ss}/{dataset}/{seed}/errors.json
        if len(parts) < 5:
            continue
        optimizer = parts[0]
        zcp_method = parts[1] if len(parts) >= 6 else None
        search_space = parts[-4]
        dataset = parts[-3]
        seed = parts[-2]
        files.append(
            {
                "path": full_path,
                "optimizer": optimizer,
                "zcp_method": zcp_method,
                "search_space": search_space,
                "dataset": dataset,
                "seed": seed,
            }
        )
    return tuple(files)


def load_run_anytime(filepath: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Build anytime curves consistent with the other table script:
    - time = cumulative search cost + eval time
    - handle two-stage (loss == -1 during stage1) and random_search (all -1) like in the other script
    Returns dict with keys: times, train, valid (all numpy arrays, times in seconds).
    Accuracies are normalized to [0,1] if needed.
    """
    try:
        with open(filepath, "r") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to read {filepath}: {exc}", file=sys.stderr)
        return None

    required = [
        "queried_train_acc",
        "queried_val_acc",
        "scaled_queried_train_time",
        "runtime",
    ]
    if not all(k in data for k in required):
        return None

    train_acc = _normalize_accuracy_scale(np.asarray(data["queried_train_acc"], dtype=float))
    valid_acc = _normalize_accuracy_scale(np.asarray(data["queried_val_acc"], dtype=float))
    eval_time = np.asarray(data["scaled_queried_train_time"], dtype=float)
    runtime = np.asarray(data["runtime"], dtype=float)

    if not (
        train_acc.size
        and train_acc.size == valid_acc.size == eval_time.size == runtime.size
    ):
        return None

    loss = np.asarray(data.get("train_loss", []), dtype=float)
    if loss.size and loss.size != runtime.size:
        loss = np.asarray([])

    is_two_stage = bool(loss.size) and np.any(loss == -1) and np.any(loss != -1)
    is_random_search = bool(loss.size) and np.all(loss == -1)

    if is_random_search:
        # No explicit runtime; use cumulative eval time
        times = np.cumsum(eval_time)
    elif is_two_stage:
        stage2_indices = np.where(loss != -1)[0]
        if stage2_indices.size == 0:
            return None
        first_stage2_idx = stage2_indices[0]
        stage1_search_cost = runtime[:first_stage2_idx].sum()
        stage2_runtime = runtime[stage2_indices]
        stage2_cumulative = np.cumsum(stage2_runtime)

        # Filter arrays to stage2 only (we still add stage1 cost offset)
        train_acc = train_acc[stage2_indices]
        valid_acc = valid_acc[stage2_indices]
        eval_time = eval_time[stage2_indices]

        times = stage1_search_cost + stage2_cumulative + eval_time
    else:
        cumulative_search_cost = np.cumsum(runtime)
        times = cumulative_search_cost + eval_time

    # Clean invalids and ensure increasing times
    m = np.isfinite(times) & np.isfinite(train_acc) & np.isfinite(valid_acc)
    if not np.any(m):
        return None
    times = times[m]
    train_acc = train_acc[m]
    valid_acc = valid_acc[m]

    order = np.argsort(times)
    times = times[order]
    train_acc = train_acc[order]
    valid_acc = valid_acc[order]

    if times.size == 0:
        return None

    return {"times": times, "train": train_acc, "valid": valid_acc}


def incumbent_curve_up_to_T(times: np.ndarray, acc: np.ndarray, T: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build incumbent staircase up to time T (inclusive).
    Returns xs, ys where ys is non-decreasing max-so-far at xs.
    """
    mask = times <= T
    if not np.any(mask):
        # No point before T; return a degenerate at t=0
        return np.array([0.0, T]), np.array([0.0, 0.0])

    xs = times[mask]
    ys = np.maximum.accumulate(acc[mask])
    # prepend (0, 0.0) to avoid area before first observation
    xs_full = np.concatenate([[0.0], xs])
    ys_full = np.concatenate([[0.0], ys])
    # append (T, last_y) to close the interval
    xs_full = np.concatenate([xs_full, [T]])
    ys_full = np.concatenate([ys_full, [ys[-1]]])
    return xs_full, ys_full


def auc_over_T(times: np.ndarray, acc: np.ndarray, T: float) -> float:
    """
    Compute AUC/T for incumbent curve on [0, T] using trapezoidal integration.
    Acc is in fractional units; return in same units (divide by T).
    """
    xs, ys = incumbent_curve_up_to_T(times, acc, T)
    if T <= 0:
        return np.nan
    area = np.trapz(ys, xs)
    return float(area / T)


def best_within_T(times: np.ndarray, acc: np.ndarray, T: float) -> float:
    mask = times <= T
    if not np.any(mask):
        return np.nan
    return float(np.nanmax(acc[mask]))


def write_output(df: pd.DataFrame, output_path: str) -> None:
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        df.to_csv(output_path, index=False)
    else:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="fixed_time", index=False)
    print(f"Wrote summary with {len(df)} rows to {output_path}")


def _format_pm(mean: float, std: float, decimals: int) -> str:
    if np.isnan(mean) or np.isnan(std):
        return r"\textemdash"
    formatted = f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"
    return f"${formatted}$"


def write_latex_table(df: pd.DataFrame, output_path: str, fractional: bool, times_by_dataset: Dict[str, float]) -> None:
    if df.empty:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    acc_unit = "" if fractional else r"\%"
    acc_header_suffix = f" ({acc_unit})" if acc_unit else ""

    columns = [
        ("Optimizer", "optimizer"),
        ("Dataset", "dataset"),
        ("ZC Proxy", "zcp_method"),
        (f"Best@T Train Acc{acc_header_suffix}", "best_T_train_mean", "best_T_train_std"),
        (f"Best@T Val Acc{acc_header_suffix}", "best_T_valid_mean", "best_T_valid_std"),
        (f"Mean@T AUC Train Acc{acc_header_suffix}", "auc_train_mean", "auc_train_std"),
        (f"Mean@T AUC Val Acc{acc_header_suffix}", "auc_valid_mean", "auc_valid_std"),
    ]
    col_spec = "lllcccc"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\rotatebox{90}{",
        r"\begin{adjustbox}{max width=\textheight}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(col[0] for col in columns) + r" \\",
        r"\midrule",
    ]

    for _, row in df.iterrows():
        formatted = []
        for col in columns:
            header = col[0]
            key = col[1]
            if len(col) == 2:
                raw_value = row[key]
                if key == "zcp_method" and (pd.isna(raw_value) or raw_value == ""):
                    raw_value = "None"
                elif pd.isna(raw_value):
                    raw_value = ""
                formatted.append(latex_escape(str(raw_value)))
            else:
                mean = float(row[col[1]])
                std = float(row[col[2]])
                decimals = 4 if fractional else 2
                formatted.append(_format_pm(mean, std, decimals))
        lines.append(" & ".join(formatted) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"}",
        ]
    )

    # Caption mentions T per dataset (seconds)
    parts = [f"{ds}: T = {int(t)} s" for ds, t in sorted(times_by_dataset.items())]
    cap = (
            "Fixed-time comparison at the maximum common budget T per dataset (for one-shot methods cumulative search cost + final evaluation; for random search cumulative evaluation cost). "
            + "; ".join(parts)
            + ". Random Search uses T+48h only for Best@T; AUC is computed at the common T for all methods. "
            + "Metrics are mean $\\pm$ std over three seeds. AUC is the incumbent mean accuracy over [0, T]."
        )

    lines.extend(
        [
            rf"\caption{{{cap}}}",
            r"\label{tab:fixed_time_final_runs}",
            r"\end{table}",
        ]
    )

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote LaTeX table to {output_path}")


def main() -> None:
    args = parse_args()
    files = find_error_files(args.root_dir)
    if not files:
        print("No errors.json files found.", file=sys.stderr)
        return

    # Load anytime curves per run
    runs = []
    for meta in files:
        curve = load_run_anytime(meta["path"])
        if curve is None:
            continue
        runs.append(
            {
                "optimizer": meta["optimizer"],
                "zcp_method": meta.get("zcp_method") or "",
                "dataset": meta["dataset"],
                "search_space": meta["search_space"],
                "seed": meta["seed"],
                "times": curve["times"],
                "train": curve["train"],
                "valid": curve["valid"],
            }
        )

    if not runs:
        print("No valid runs found.", file=sys.stderr)
        return

    # Determine T per dataset: minimum of last time across all runs in that dataset
    times_by_dataset: Dict[str, float] = {}
    for dataset in sorted({r["dataset"] for r in runs}):
        last_times = [float(r["times"][-1]) for r in runs if r["dataset"] == dataset and r["times"].size > 0]
        if not last_times:
            continue
        T = float(np.min(last_times))
        if T <= 0:
            continue
        times_by_dataset[dataset] = T

    if not times_by_dataset:
        print("Could not determine common time T for any dataset.", file=sys.stderr)
        return

    # Aggregate per (optimizer, zcp_method, dataset, search_space)
    grouped = defaultdict(list)
    for r in runs:
        ds = r["dataset"]
        if ds not in times_by_dataset:
            continue
        T_common = times_by_dataset[ds]

        # Best@T: allow +48h for Random Search
        T_best = T_common + (EXTRA_T_RANDOM_SEARCH if r["optimizer"] == "random_search" else 0.0)
        # AUC@T: always use the common T for fairness
        T_auc = T_common

        best_train = best_within_T(r["times"], r["train"], T_best)
        best_valid = best_within_T(r["times"], r["valid"], T_best)
        auc_train = auc_over_T(r["times"], r["train"], T_auc)
        auc_valid = auc_over_T(r["times"], r["valid"], T_auc)
        grouped[(r["optimizer"], r["zcp_method"], ds, r["search_space"])].append(
            {
                "best_T_train": best_train,
                "best_T_valid": best_valid,
                "auc_train": auc_train,
                "auc_valid": auc_valid,
            }
        )

    factor = 1.0 if args.fractional else 100.0
    rows = []
    for (optimizer, zcp_method, dataset, search_space), metrics in sorted(grouped.items()):
        best_T_train_vals = np.array([m["best_T_train"] for m in metrics], dtype=float) * factor
        best_T_valid_vals = np.array([m["best_T_valid"] for m in metrics], dtype=float) * factor
        auc_train_vals = np.array([m["auc_train"] for m in metrics], dtype=float) * factor
        auc_valid_vals = np.array([m["auc_valid"] for m in metrics], dtype=float) * factor

        rows.append(
            {
                "optimizer": optimizer,
                "dataset": dataset,
                "zcp_method": zcp_method,
                "best_T_train_mean": float(np.nanmean(best_T_train_vals)),
                "best_T_train_std": float(np.nanstd(best_T_train_vals, ddof=0)),
                "best_T_valid_mean": float(np.nanmean(best_T_valid_vals)),
                "best_T_valid_std": float(np.nanstd(best_T_valid_vals, ddof=0)),
                "auc_train_mean": float(np.nanmean(auc_train_vals)),
                "auc_train_std": float(np.nanstd(auc_train_vals, ddof=0)),
                "auc_valid_mean": float(np.nanmean(auc_valid_vals)),
                "auc_valid_std": float(np.nanstd(auc_valid_vals, ddof=0)),
            }
        )

    if not rows:
        print("No metrics to summarize.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    df.sort_values(["dataset", "optimizer", "zcp_method"], inplace=True)
    write_output(df, args.output)

    latex_output = args.latex_output or os.path.splitext(os.path.abspath(args.output))[0] + "_table.txt"
    write_latex_table(df, latex_output, args.fractional, times_by_dataset)

    # Console hint for chosen T per dataset
    for ds, T in sorted(times_by_dataset.items()):
        print(f"[INFO] Dataset '{ds}': chosen common time T = {T:.2f} s")


if __name__ == "__main__":
    main()