import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

DATASETS = {"cifar10", "cifar100", "ImageNet16-120"}

ZCP_PRE_METHODS = {
    "zcp-pre_gsparsity",
    "zcp-pre_zcp_gsparsity",
}

# Extra time budget for Random Search (in seconds)
SECONDS_IN_48H = 48 * 3600  # 48 hours
# Tiny epsilon to nudge T so boundary points are included by <= T
EPS_REL = 1e-9
EPS_ABS = 1e-6


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
    # NEW: durations for zcp-pre shift
    parser.add_argument(
        "--durations_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor",
        help="Directory containing arch_scores_duration_*.json files for zcp-pre shift.",
    )
    # NEW: random_search extra time controls
    parser.add_argument(
        "--rs_extra_time",
        type=float,
        default=SECONDS_IN_48H,
        help="Additional budget (in seconds) granted to random_search for Best@T unless overridden.",
    )
    parser.add_argument(
        "--rs_extra_time_override",
        type=str,
        nargs="*",
        default=[],
        help="Per-dataset overrides as DATASET=SECONDS (e.g., cifar10=7200 ImageNet16-120=14400).",
    )
    return parser.parse_args()


def load_zcp_pre_durations(durations_dir: str) -> Dict[str, float]:
    """
    Load per-dataset one-time durations for zcp-pre methods from durations_dir.
    Expects: arch_scores_duration_<DATASET>.json with schema: {"duration": <float>}
    """
    durations: Dict[str, float] = {}
    for ds in DATASETS:
        fname = f"arch_scores_duration_{ds}.json"
        fpath = os.path.join(durations_dir, fname)
        try:
            with open(fpath, "r") as fh:
                data = json.load(fh)
            durations[ds] = float(data.get("duration", 0.0))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
    return durations


def parse_rs_extra_overrides(pairs: List[str]) -> Dict[str, float]:
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


def dataset_random_guess(dataset: str) -> float:
    if dataset == "cifar10":
        return 1.0 / 10.0
    if dataset == "cifar100":
        return 1.0 / 100.0
    if dataset == "ImageNet16-120":
        return 1.0 / 120.0
    return 0.0


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

    train_acc = _normalize_accuracy_scale(
        np.asarray(data["queried_train_acc"], dtype=float)
    )
    valid_acc = _normalize_accuracy_scale(
        np.asarray(data["queried_val_acc"], dtype=float)
    )
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


def incumbent_curve_up_to_T(
    times: np.ndarray, acc: np.ndarray, T: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build incumbent staircase up to time T (inclusive) and hold last value constant to T.
    Returns xs, ys where ys is non-decreasing max-so-far at xs.
    """
    mask = times <= T
    if not np.any(mask):
        return np.array([0.0, T]), np.array([0.0, 0.0])

    xs = times[mask]
    ys = np.maximum.accumulate(acc[mask])
    xs_full = np.concatenate([[0.0], xs, [T]])
    ys_full = np.concatenate([[0.0], ys, [ys[-1]]])  # constant after last
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


# New helpers for stability band area (upper-lower envelope)
def _dedup_sorted_times(
    times: np.ndarray, values: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure strictly increasing times by merging duplicates (keep last value for duplicate timestamps).
    Returns sorted times and corresponding values.
    """
    order = np.argsort(times)
    t = times[order]
    v = values[order]
    if t.size == 0:
        return t, v
    t_new = [t[0]]
    v_new = [v[0]]
    for ti, vi in zip(t[1:], v[1:]):
        if abs(ti - t_new[-1]) <= eps:
            # overwrite last value for duplicate time
            t_new[-1] = ti
            v_new[-1] = vi
        else:
            t_new.append(ti)
            v_new.append(vi)
    return np.asarray(t_new, dtype=float), np.asarray(v_new, dtype=float)


def _segment_slopes(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    if times.size < 2:
        return np.asarray([], dtype=float)
    dt = np.diff(times)
    dv = np.diff(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = np.where(dt != 0.0, dv / dt, 0.0)
    return slopes


def _line_coeff_on_interval(
    times: np.ndarray, values: np.ndarray, slopes: np.ndarray, t_mid: float
) -> Tuple[float, float]:
    """
    For a given midpoint t_mid (assumed to be in an interval without internal breakpoints),
    return (a, b) for y = a*t + b of the seed's piecewise-linear function at that interval,
    using linear extrapolation before the first point and after the last point if needed.
    """
    n = times.size
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        # Constant function
        return 0.0, float(values[0])
    if t_mid <= times[0]:
        a = float(slopes[0])
        b = float(values[0] - a * times[0])
        return a, b
    if t_mid >= times[-1]:
        a = float(slopes[-1])
        b = float(values[-1] - a * times[-1])
        return a, b
    # find segment idx so that times[idx] <= t_mid <= times[idx+1]
    idx = np.searchsorted(times, t_mid, side="right") - 1
    idx = int(np.clip(idx, 0, n - 2))
    a = float(slopes[idx])
    b = float(values[idx] - a * times[idx])
    return a, b


def _eval_piecewise_linear(
    times: np.ndarray, values: np.ndarray, t_eval: np.ndarray
) -> np.ndarray:
    """
    Evaluate a piecewise-linear curve at t_eval using linear interpolation inside
    and constant extrapolation outside [times[0], times[-1]] (hold first/last values).
    """
    n = times.size
    if n == 0:
        return np.full_like(t_eval, np.nan, dtype=float)
    if n == 1:
        return np.full_like(t_eval, float(values[0]), dtype=float)
    # linear interpolation; constant outside
    return np.interp(
        t_eval, times, values, left=float(values[0]), right=float(values[-1])
    )


def band_area_over_T(seeds: List[Dict[str, np.ndarray]], key: str, T: float) -> float:
    """
    Compute average band width (1/T' * ∫_{t0}^{T} [max f_s(t) - min f_s(t)] dt)
    where t0 = max first timestamp across seeds clipped to T. Curves are linearly
    interpolated between observed points and held constant outside their range.
    T' = (T - t0); returns value in same units as accuracy (fractional).
    """
    if T <= 0:
        return np.nan
    if not seeds:
        return np.nan

    # Prepare per-seed cleaned curves
    curves = []
    firsts = []
    for s in seeds:
        t = np.asarray(s["times"], dtype=float)
        y = np.asarray(s[key], dtype=float)
        m = np.isfinite(t) & np.isfinite(y) & (t >= 0.0)
        t, y = t[m], y[m]
        t, y = _dedup_sorted_times(t, y)
        if t.size == 0:
            continue
        curves.append((t, y, _segment_slopes(t, y)))
        firsts.append(float(t[0]))
    if len(curves) == 0:
        return np.nan
    if len(curves) == 1:
        return 0.0

    # Start integration at latest first timestamp among seeds (clipped to T)
    t0 = min(T, float(np.max(firsts)))
    if not np.isfinite(t0) or t0 >= T:
        return 0.0
    Tprime = T - t0

    # Base grid: t0, T, and all seed timestamps in [t0, T]
    grid = {float(t0), float(T)}
    for t, _, _ in curves:
        t_clip = t[(t >= t0) & (t <= T)]
        grid.update(map(float, t_clip.tolist()))
    g0 = np.array(sorted(grid), dtype=float)
    if g0.size < 2:
        return 0.0

    # Add pairwise intersections of lines on each base interval
    intersections = []
    for i in range(g0.size - 1):
        left, right = g0[i], g0[i + 1]
        if right - left <= 0.0:
            continue
        t_mid = 0.5 * (left + right)
        coeffs = [_line_coeff_on_interval(t, y, s, t_mid) for (t, y, s) in curves]
        n = len(coeffs)
        for p in range(n):
            a1, b1 = coeffs[p]
            for q in range(p + 1, n):
                a2, b2 = coeffs[q]
                denom = a1 - a2
                if abs(denom) <= 1e-18:
                    continue
                t_cross = (b2 - b1) / denom
                if left < t_cross < right:
                    intersections.append(float(t_cross))
    g = (
        np.array(sorted(set(g0.tolist() + intersections)), dtype=float)
        if intersections
        else g0
    )

    # Evaluate all seeds with constant extrapolation outside their range
    Y = []
    for t, y, _ in curves:
        Y.append(_eval_piecewise_linear(t, y, g))
    Y = np.vstack(Y)
    gap = np.nanmax(Y, axis=0) - np.nanmin(Y, axis=0)

    area = np.trapz(gap, g)
    return float(area / Tprime)


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


def write_latex_table(
    df: pd.DataFrame,
    output_path: str,
    fractional: bool,
    times_by_dataset: Dict[str, float],
    rs_extra_by_dataset: Dict[str, float],  # NEW
) -> None:
    if df.empty:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    acc_unit = "" if fractional else r"\%"
    acc_header_suffix = f" ({acc_unit})" if acc_unit else ""

    columns = [
        ("Optimizer", "optimizer"),
        ("Dataset", "dataset"),
        ("ZC Proxy", "zcp_method"),
        (
            f"Best@T Train Acc{acc_header_suffix}",
            "best_T_train_mean",
            "best_T_train_std",
        ),
        (f"Best@T Val Acc{acc_header_suffix}", "best_T_valid_mean", "best_T_valid_std"),
        (f"Mean@T AUC Train Acc{acc_header_suffix}", "auc_train_mean", "auc_train_std"),
        (f"Mean@T AUC Val Acc{acc_header_suffix}", "auc_valid_mean", "auc_valid_std"),
        (f"Band Area Train Acc{acc_header_suffix}", "band_area_train"),
        (f"Band Area Val Acc{acc_header_suffix}", "band_area_valid"),
    ]
    col_spec = "lllcccccc"

    numeric_single_cols = {"band_area_train", "band_area_valid"}

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
                if key in numeric_single_cols and raw_value != "":
                    decimals = 4 if fractional else 2
                    formatted.append(f"${float(raw_value):.{decimals}f}$")
                else:
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

    # Caption includes T per dataset and RS extra time per dataset (seconds)
    parts_T = [f"{ds}: T = {int(t)} s" for ds, t in sorted(times_by_dataset.items())]
    parts_RS = [
        f"{ds}: +{int(rs_extra_by_dataset.get(ds, 0.0))} s"
        for ds in sorted(times_by_dataset)
    ]
    cap = (
        "Fixed-time comparison at the maximum common budget T per dataset "
        "(T = max(latest first cumulative time, earliest end) + $\\epsilon$; "
        "for one-shot methods cumulative search cost + final evaluation; for random search cumulative evaluation cost). "
        + "; ".join(parts_T)
        + ". Random Search uses T plus a dataset-specific extra time only for Best@T: "
        + "; ".join(parts_RS)
        + ". AUC is computed at the common T for all methods. "
        + "Metrics are mean $\\pm$ std over three seeds. AUC uses an incumbent that is held constant after the last observation up to T. "
        + "Mean Band Gap integrates from the latest first timestamp among seeds to T, with curves held constant outside their observed range."
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


def _make_unique_path(path: str) -> str:
    """
    If path exists, append _1, _2, ... before the extension until a free name is found.
    """
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    i = 1
    while True:
        candidate = f"{root}_{i}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def compute_and_save_ranks(df: pd.DataFrame, base_output_path: str) -> None:
    """
    Create a per-dataset ranking CSV for:
      - mean Best@T Val Acc (higher is better)
      - Mean@T Val AUC (higher is better)
      - Val bandwidth = band_area_valid (lower is better)
    Also prints human-readable rankings per dataset.
    """
    if df.empty:
        print("[RANK] No data to rank.")
        return

    # Ensure required columns exist
    required_cols = {
        "optimizer",
        "dataset",
        "zcp_method",
        "best_T_valid_mean",
        "auc_valid_mean",
        "band_area_valid",
    }
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[RANK] Missing columns for ranking: {sorted(missing)}", file=sys.stderr)
        return

    # Round numeric columns to two decimals BEFORE ranking
    round_cols = ["best_T_valid_mean", "auc_valid_mean", "band_area_valid"]
    df = df.copy()
    for c in round_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    rank_rows = []
    for ds, sub in df.groupby("dataset", sort=True):
        sub = sub.copy()
        n = len(sub)

        # Ranks (NaNs get worst rank n+1)
        r_best = sub["best_T_valid_mean"].rank(method="min", ascending=False)
        r_auc = sub["auc_valid_mean"].rank(method="min", ascending=False)
        r_band = sub["band_area_valid"].rank(method="min", ascending=True)

        sub["rank_best_T_valid_mean"] = r_best.fillna(n + 1).astype(int)
        sub["rank_auc_valid_mean"] = r_auc.fillna(n + 1).astype(int)
        sub["rank_band_area_valid"] = r_band.fillna(n + 1).astype(int)

        # Print summaries (optional: force two-decimal display)
        print(f"\n[RANK] Dataset: {ds} — mean Best@T Val Acc (higher is better)")
        print(
            sub.sort_values(["rank_best_T_valid_mean", "optimizer", "zcp_method"])[
                [
                    "rank_best_T_valid_mean",
                    "optimizer",
                    "zcp_method",
                    "best_T_valid_mean",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
        )

        print(f"\n[RANK] Dataset: {ds} — Mean@T Val AUC (higher is better)")
        print(
            sub.sort_values(["rank_auc_valid_mean", "optimizer", "zcp_method"])[
                ["rank_auc_valid_mean", "optimizer", "zcp_method", "auc_valid_mean"]
            ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
        )

        print(f"\n[RANK] Dataset: {ds} — Val bandwidth (band area; lower is better)")
        print(
            sub.sort_values(["rank_band_area_valid", "optimizer", "zcp_method"])[
                ["rank_band_area_valid", "optimizer", "zcp_method", "band_area_valid"]
            ].to_string(index=False, float_format=lambda x: f"{x:.2f}")
        )

        rank_rows.append(
            sub[
                [
                    "optimizer",
                    "dataset",
                    "zcp_method",
                    "rank_best_T_valid_mean",
                    "rank_auc_valid_mean",
                    "rank_band_area_valid",
                    "best_T_valid_mean",
                    "auc_valid_mean",
                    "band_area_valid",
                ]
            ]
        )

    ranks_df = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()

    # Ensure two-decimal precision in the saved CSV as well
    for c in round_cols:
        if c in ranks_df.columns:
            ranks_df[c] = pd.to_numeric(ranks_df[c], errors="coerce").round(2)

    # Safe output path next to base output
    out_dir = os.path.dirname(os.path.abspath(base_output_path))
    os.makedirs(out_dir, exist_ok=True)
    base_root, _ = os.path.splitext(os.path.basename(base_output_path))
    default_path = os.path.join(out_dir, f"{base_root}_ranks.csv")
    safe_path = _make_unique_path(default_path)

    ranks_df.to_csv(safe_path, index=False)
    print(f"\n[RANK] Wrote ranks CSV to {safe_path}")


def _normalize_opt_key(name: str) -> str:
    """
    Normalize optimizer names by removing 'zcp' tokens and non-alphanumerics to
    allow flexible matching between base and zcp-variants.
    """
    if name is None:
        return ""
    s = str(name).lower()
    # remove the specific token 'zcp' (handles 'zcp-', 'zcp_', 'zcp' etc.)
    s = s.replace("zcp", "")
    # remove non-alphanumeric characters
    s = "".join(ch for ch in s if ch.isalnum())
    return s


def _match_optimizers(df_opts: List[str], query: str) -> List[str]:
    """
    Return list of optimizer names from df_opts that match the query according to
    normalized-substring rule. This lets every zcp variant that contains the same
    base token participate in comparisons.
    """
    norm_q = _normalize_opt_key(query)
    matches = []
    for o in df_opts:
        norm_o = _normalize_opt_key(o)
        if not norm_q:
            continue
        # match either way to be permissive
        if norm_q in norm_o or norm_o in norm_q:
            matches.append(o)
    return sorted(set(matches))


ZCP_METHODS = {"jacov", "params", "synflow"}


def _is_zcp_optimizer(name: str) -> bool:
    """
    Treat any optimizer name that contains 'zcp' as zcp-scaled.
    E.g., 'zcp_gsparsity', 'pre_zcp_gsparsity', 'zcp-pre_zcp_gsparsity'.
    """
    if name is None:
        return False
    return "zcp" in str(name).lower()


def _cmp_scalar_rounded(
    a: float, b: float, higher_is_better: bool, decimals: int = 2
) -> int:
    """
    Round a and b to 'decimals' and compare strictly with no tolerance.
    +1 if a wins, -1 if a loses, 0 if equal. NaNs are treated as worst.
    """
    if pd.isna(a) and pd.isna(b):
        return 0
    if pd.isna(a):
        return -1
    if pd.isna(b):
        return 1
    ar = round(float(a), decimals)
    br = round(float(b), decimals)
    if ar == br:
        return 0
    if higher_is_better:
        return 1 if ar > br else -1
    else:
        return 1 if ar < br else -1


def _collect_side_rows(
    df: pd.DataFrame,
    ds: str,
    left_token: str,
    mode: str,
) -> pd.DataFrame:
    """
    Collect rows for one side for a given dataset 'ds' based on mode:
      - mode == 'optimizer': left_token is an optimizer name.
        If the optimizer has zcp variants (zcp_method in ZCP_METHODS), keep only those;
        else keep all rows (non-scaled).
      - mode == 'zcp_method': left_token is in ZCP_METHODS; select rows by that zcp_method.
    """
    sub = df[df["dataset"] == ds]
    if mode == "optimizer":
        side = sub[sub["optimizer"] == left_token]
        if side["zcp_method"].isin(ZCP_METHODS).any():
            side = side[side["zcp_method"].isin(ZCP_METHODS)]
        return side
    elif mode == "zcp_method":
        return sub[sub["zcp_method"] == left_token]
    else:
        return sub.iloc[0:0]


def compute_pairwise_scores(
    df: pd.DataFrame,
    comparisons: List[Tuple[str, str]],
    metric_col: str,
    datasets: List[str],
    higher_is_better: bool,
) -> pd.DataFrame:
    """
    Compute integer scores per dataset for requested comparisons.
    Rules:
      - Round both sides to two decimals before comparing, no tolerance.
      - +1 win, -1 loss, 0 tie. Higher wins for AUC; lower wins for bandwidth.
      - If both sides are zcp-optimizers (contain 'zcp'), compare every zcp variant vs every zcp variant.
      - If one side is non-zcp, every zcp variant on the other side competes against each non-zcp row (many-vs-one).
      - For 'jacov vs synflow' (and similar), compare by zcp_method but ONLY across zcp-optimizers that have BOTH methods
        present on that dataset (optimizer intersection). This yields pairs like:
          {zcp_gsparsity|jacov, zcp-pre_zcp_gsparsity|jacov} vs {zcp_gsparsity|synflow, zcp-pre_zcp_gsparsity|synflow}.
      - If one side has zero rows and the other has n rows, score is -n for the empty side (and +n for the non-empty side).
    """
    # Ensure metric exists and is numeric; do not modify df values
    if metric_col not in df.columns:
        print(
            f"[PAIRWISE] Missing column '{metric_col}' for comparison.", file=sys.stderr
        )
        return pd.DataFrame(columns=["comparison"] + datasets + ["sum"])

    base = df.copy()
    base[metric_col] = pd.to_numeric(base[metric_col], errors="coerce")

    rows = []
    for a_name, b_name in comparisons:
        mode = (
            "zcp_method"
            if (a_name in ZCP_METHODS and b_name in ZCP_METHODS)
            else "optimizer"
        )

        row = {"comparison": f"{a_name} vs {b_name}"}
        total = 0
        for ds in datasets:
            if mode == "zcp_method":
                sub = base[base["dataset"] == ds]
                left_opts = set(
                    sub.loc[sub["zcp_method"] == a_name, "optimizer"]
                    .dropna()
                    .astype(str)
                )
                right_opts = set(
                    sub.loc[sub["zcp_method"] == b_name, "optimizer"]
                    .dropna()
                    .astype(str)
                )
                # Optimizers present with BOTH methods (data-driven)
                allowed_opts = left_opts & right_opts
                a_rows = sub[
                    (sub["zcp_method"] == a_name)
                    & (sub["optimizer"].isin(list(allowed_opts)))
                ]
                b_rows = sub[
                    (sub["zcp_method"] == b_name)
                    & (sub["optimizer"].isin(list(allowed_opts)))
                ]
            else:
                a_rows = _collect_side_rows(base, ds, a_name, mode)
                b_rows = _collect_side_rows(base, ds, b_name, mode)

            a_vals = (
                a_rows[metric_col].astype(float).values
                if not a_rows.empty
                else np.array([], dtype=float)
            )
            b_vals = (
                b_rows[metric_col].astype(float).values
                if not b_rows.empty
                else np.array([], dtype=float)
            )

            if a_vals.size == 0 and b_vals.size == 0:
                score = 0
            elif a_vals.size == 0:
                score = -int(b_vals.size)
            elif b_vals.size == 0:
                score = +int(a_vals.size)
            else:
                s = 0
                for va in a_vals:
                    for vb in b_vals:
                        s += _cmp_scalar_rounded(
                            va, vb, higher_is_better=higher_is_better, decimals=2
                        )
                score = int(s)

            row[ds] = score
            total += score

        row["sum"] = int(total)
        rows.append(row)

    return pd.DataFrame(rows, columns=["comparison"] + datasets + ["sum"])


def write_pairwise_latex(df_table: pd.DataFrame, out_path: str, caption: str) -> None:
    """
    Emit a compact LaTeX table with sign-formatted integers per dataset and a Sum column.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    datasets = [c for c in df_table.columns if c not in ("comparison", "sum")]
    col_spec = "l" + "r" * (len(datasets) + 1)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Comparison & " + " & ".join(datasets) + " & Sum \\\\",
        r"\midrule",
    ]
    for _, r in df_table.iterrows():
        cells = [latex_escape(str(r["comparison"]))]
        for ds in datasets:
            val = int(r[ds])
            cells.append(f"{val:+d}")
        cells.append(f"{int(r['sum']):+d}")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}"]
    )
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[PAIRWISE] Wrote LaTeX table to {out_path}")


def main() -> None:
    args = parse_args()

    # NEW: load zcp-pre durations and RS extra-time overrides
    durations_map = load_zcp_pre_durations(args.durations_dir)
    rs_overrides = parse_rs_extra_overrides(args.rs_extra_time_override)

    files = find_error_files(args.root_dir)
    if not files:
        print("No errors.json files found.", file=sys.stderr)
        return

    # Load anytime curves per run and apply zcp-pre shift to times
    runs = []
    for meta in files:
        curve = load_run_anytime(meta["path"])
        if curve is None:
            continue

        times = curve["times"]
        train = curve["train"]
        valid = curve["valid"]

        # Apply one-time dataset-specific shift for zcp-pre methods
        optimizer = meta["optimizer"]
        dataset = meta["dataset"]
        if optimizer in ZCP_PRE_METHODS:
            offset = float(durations_map.get(dataset, 0.0))
            if np.isfinite(offset) and offset > 0:
                times = times + offset  # shift to the right

        runs.append(
            {
                "optimizer": optimizer,
                "zcp_method": meta.get("zcp_method") or "",
                "dataset": dataset,
                "search_space": meta["search_space"],
                "seed": meta["seed"],
                "times": times,
                "train": train,
                "valid": valid,
            }
        )

    if not runs:
        print("No valid runs found.", file=sys.stderr)
        return

    # Determine T per dataset on shifted curves (without injecting baseline)
    times_by_dataset: Dict[str, float] = {}
    coverage_debug: Dict[str, Tuple[float, float]] = {}
    for dataset in sorted({r["dataset"] for r in runs}):
        ds_runs = [r for r in runs if r["dataset"] == dataset and r["times"].size > 0]
        if not ds_runs:
            continue
        first_times = [float(r["times"][0]) for r in ds_runs]
        last_times = [float(r["times"][-1]) for r in ds_runs]
        T_start = float(np.max(first_times))  # latest first observation across runs
        T_end = float(np.min(last_times))  # earliest end across runs
        T_base = max(T_start, T_end)
        if T_base <= 0:
            continue
        eps = max(EPS_ABS, EPS_REL * T_base)
        T = T_base + eps
        times_by_dataset[dataset] = T
        coverage_debug[dataset] = (T_start, T_end)

    if not times_by_dataset:
        print("Could not determine common time T for any dataset.", file=sys.stderr)
        return

    # Build dataset-specific RS extra time map
    ds_in_runs = sorted(times_by_dataset.keys())
    rs_extra_by_dataset = {
        ds: rs_overrides.get(ds, args.rs_extra_time) for ds in ds_in_runs
    }

    # Helper to inject baseline at t=0 for AUC/Band (without affecting Best@T or T determination)
    def inject_baseline(
        times: np.ndarray, acc: np.ndarray, baseline: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        t = np.asarray(times, dtype=float)
        a = np.asarray(acc, dtype=float)
        if t.size == 0:
            return np.array([0.0], dtype=float), np.array([baseline], dtype=float)
        if t[0] > 0.0:
            t = np.concatenate([[0.0], t])
            a = np.concatenate([[baseline], a])
        elif t[0] == 0.0:
            a = a.copy()
            a[0] = baseline
        return t, a

    # Aggregate per (optimizer, zcp_method, dataset, search_space)
    grouped = defaultdict(list)
    runs_by_group = defaultdict(list)
    for r in runs:
        ds = r["dataset"]
        if ds not in times_by_dataset:
            continue
        T_common = times_by_dataset[ds]

        # Best@T: allow dataset-specific extra for Random Search
        extra = (
            rs_extra_by_dataset.get(ds, 0.0)
            if r["optimizer"] == "random_search"
            else 0.0
        )
        T_best = T_common + max(0.0, float(extra))
        # AUC@T: always use the common T for fairness
        T_auc = T_common

        # AUC should start from baseline at t=0 and interpolate to first shifted point
        train_baseline = 0.0
        valid_baseline = dataset_random_guess(ds)
        t_train, y_train = inject_baseline(r["times"], r["train"], train_baseline)
        t_valid, y_valid = inject_baseline(r["times"], r["valid"], valid_baseline)

        best_train = best_within_T(r["times"], r["train"], T_best)
        best_valid = best_within_T(r["times"], r["valid"], T_best)
        auc_train = auc_over_T(t_train, y_train, T_auc)
        auc_valid = auc_over_T(t_valid, y_valid, T_auc)

        key = (r["optimizer"], r["zcp_method"], ds, r["search_space"])
        grouped[key].append(
            {
                "best_T_train": best_train,
                "best_T_valid": best_valid,
                "auc_train": auc_train,
                "auc_valid": auc_valid,
            }
        )
        runs_by_group[key].append(r)

    factor = 1.0 if args.fractional else 100.0
    rows = []
    for (optimizer, zcp_method, dataset, search_space), metrics in sorted(
        grouped.items()
    ):
        best_T_train_vals = (
            np.array([m["best_T_train"] for m in metrics], dtype=float) * factor
        )
        best_T_valid_vals = (
            np.array([m["best_T_valid"] for m in metrics], dtype=float) * factor
        )
        auc_train_vals = (
            np.array([m["auc_train"] for m in metrics], dtype=float) * factor
        )
        auc_valid_vals = (
            np.array([m["auc_valid"] for m in metrics], dtype=float) * factor
        )

        # Stability band area (average gap over [0, T]) for train/val
        T_common = times_by_dataset[dataset]
        seeds_orig = runs_by_group[(optimizer, zcp_method, dataset, search_space)]

        # Inject baselines for band metric (train: 0.0, valid: random guess), keeping shifted times
        seeds_aug = []
        for s in seeds_orig:
            t = s["times"]
            tr_b = 0.0
            va_b = dataset_random_guess(dataset)
            t_train, y_train = inject_baseline(t, s["train"], tr_b)
            t_valid, y_valid = inject_baseline(t, s["valid"], va_b)
            seeds_aug.append(
                {"times": t_train, "train": y_train, "valid": y_valid}
            )  # times equal in both after injection

        band_train = band_area_over_T(seeds_aug, key="train", T=T_common) * factor
        band_valid = band_area_over_T(seeds_aug, key="valid", T=T_common) * factor

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
                "band_area_train": float(band_train),
                "band_area_valid": float(band_valid),
            }
        )

    if not rows:
        print("No metrics to summarize.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    df.sort_values(["dataset", "optimizer", "zcp_method"], inplace=True)
    write_output(df, args.output)

    latex_output = (
        args.latex_output
        or os.path.splitext(os.path.abspath(args.output))[0] + "_table.txt"
    )
    write_latex_table(
        df, latex_output, args.fractional, times_by_dataset, rs_extra_by_dataset
    )

    compute_and_save_ranks(df, args.output)

    # Console hint for chosen T per dataset and coverage
    for ds, T in sorted(times_by_dataset.items()):
        T_start, T_end = coverage_debug.get(ds, (float("nan"), float("nan")))
        print(
            f"[INFO] Dataset '{ds}': chosen common time T = {T:.6f} s "
            f"(latest first = {T_start:.6f} s, earliest end = {T_end:.6f} s)"
        )

    compute_and_save_ranks(df, args.output)

    # --- NEW: Generate pairwise comparison tables ----------------------------
    comparisons = [
        ("gsparsity", "zcp-pre_gsparsity"),
        ("gsparsity", "zcp_gsparsity"),
        ("gsparsity", "zcp-pre_zcp_gsparsity"),
        ("zcp_gsparsity", "zcp-pre_gsparsity"),
        ("zcp_gsparsity", "zcp-pre_zcp_gsparsity"),
        ("pre_gsparsity", "zcp-pre_zcp_gsparsity"),
        ("jacov", "synflow"),
        ("jacov", "params"),
        ("synflow", "params"),
    ]
    ds_cols = ["cifar10", "cifar100", "ImageNet16-120"]

    # Base name for pairwise outputs next to the main LaTeX/table output
    latex_base = (
        args.latex_output
        or os.path.splitext(os.path.abspath(args.output))[0] + "_table.txt"
    )
    base_root, _ = os.path.splitext(latex_base)

    # Mean@T Val AUC (higher is better)
    if "auc_valid_mean" in df.columns:
        auc_tab = compute_pairwise_scores(
            df=df,
            comparisons=comparisons,
            metric_col="auc_valid_mean",
            datasets=ds_cols,
            higher_is_better=True,
        )
        auc_out = f"{base_root}_pairwise_auc_valid.txt"
        cap_auc = "Pairwise scores (+1 win, -1 loss, 0 tie) on Mean@T Val AUC. Values rounded to two decimals; higher is better."
        write_pairwise_latex(auc_tab, auc_out, cap_auc)

    # Val bandwidth (lower is better) — band_area_valid
    if "band_area_valid" in df.columns:
        band_tab = compute_pairwise_scores(
            df=df,
            comparisons=comparisons,
            metric_col="band_area_valid",
            datasets=ds_cols,
            higher_is_better=False,
        )
        band_out = f"{base_root}_pairwise_bandwidth.txt"
        cap_band = "Pairwise scores (+1 win, -1 loss, 0 tie) on Val bandwidth (band area). Values rounded to two decimals; lower is better."
        write_pairwise_latex(band_tab, band_out, cap_band)

    # -------------------------------------------------------------------------

    # Console hint for chosen T per dataset and coverage
    for ds, T in sorted(times_by_dataset.items()):
        # ...existing code...
        print(
            f"[INFO] Dataset '{ds}': chosen common time T = {T:.6f} s "
            f"(latest first = {T_start:.6f} s, earliest end = {T_end:.6f} s)"
        )


if __name__ == "__main__":
    main()
