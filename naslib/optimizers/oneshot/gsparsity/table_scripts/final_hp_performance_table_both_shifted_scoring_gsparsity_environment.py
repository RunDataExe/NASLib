import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

DATASET_CLASSES = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}

# NEW: zcp-pre methods to receive a one-time time addition
ZCP_PRE_METHODS = {
    "zcp-pre_gsparsity",
    "zcp-pre_zcp_gsparsity",
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate final anytime performance (train/val accuracy and time) "
            "per optimizer / dataset / zero-cost proxy."
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
        default="naslib/optimizers/oneshot/gsparsity/table_scripts/tables/final_performance_summary.xlsx",
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
        default="naslib/optimizers/oneshot/gsparsity/table_scripts/tables/final_performance_summary_table.txt",
        help="Optional path for the LaTeX table (.txt). Defaults to output base name.",
    )
    # NEW: directory with per-dataset durations to add for zcp-pre methods
    parser.add_argument(
        "--durations_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor",
        help="Directory containing arch_scores_duration_*.json files with {'duration': seconds}.",
    )
    return parser.parse_args()


def load_zcp_pre_durations(durations_dir: str) -> Dict[str, float]:
    """
    Load per-dataset one-time durations for zcp-pre methods from durations_dir.
    Expects files named: arch_scores_duration_<DATASET>.json containing: {'duration': <float>}.
    """
    durations: Dict[str, float] = {}
    for ds in DATASET_CLASSES.keys():
        fname = f"arch_scores_duration_{ds}.json"
        fpath = os.path.join(durations_dir, fname)
        try:
            with open(fpath, "r") as fh:
                data = json.load(fh)
            durations[ds] = float(data.get("duration", 0.0))
        except (OSError, json.JSONDecodeError, ValueError):
            # Silently skip missing/invalid files
            continue
    return durations


def find_error_files(root_dir: str) -> Tuple[Dict, ...]:
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "errors.json" not in filenames:
            continue
        full_path = os.path.join(dirpath, "errors.json")
        relative_path = os.path.relpath(full_path, root_dir)
        parts = relative_path.split(os.sep)
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


def compute_run_stats(
    filepath: str, dataset: str, optimizer: str, durations_map: Dict[str, float]
) -> Optional[Dict[str, float]]:
    try:
        with open(filepath, "r") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to read {filepath}: {exc}", file=sys.stderr)
        return None

    opt_lower = (optimizer or "").lower()
    is_gsparsity_variant = ("gsparsity" in opt_lower) or ("gsparstity" in opt_lower)

    if is_gsparsity_variant:
        # Use raw train_acc / valid_acc and cumulative runtime only.
        required = ["train_acc", "valid_acc", "runtime"]
        if not all(k in data for k in required):
            return None

        # Support list or scalar
        def _to_array(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                arr = np.asarray(x, dtype=float)
            else:
                try:
                    arr = np.asarray([float(x)], dtype=float)
                except Exception:
                    arr = np.asarray([], dtype=float)
            return arr

        train_acc_arr = _to_array(data.get("train_acc", []))
        valid_acc_arr = _to_array(data.get("valid_acc", []))
        runtime_arr = _to_array(data.get("runtime", []))

        n = min(train_acc_arr.size, valid_acc_arr.size, runtime_arr.size)
        if n == 0:
            return None

        train_acc_series = train_acc_arr[:n]
        valid_acc_series = valid_acc_arr[:n]
        runtime_series = runtime_arr[:n]

        # cumulative time of training epochs (no scaled train time)
        total_time = np.cumsum(runtime_series)
        if total_time.size == 0:
            return None

        final_time = float(total_time[-1])
        final_train = float(train_acc_series[-1])  # last observed value only
        final_valid = float(valid_acc_series[-1])  # last observed value only

        # Keep the one-time shift for zcp-pre methods
        if optimizer in ZCP_PRE_METHODS:
            extra = float(durations_map.get(dataset, 0.0))
            if np.isfinite(extra) and extra > 0:
                final_time += extra

        return {
            "train_acc": final_train,
            "valid_acc": final_valid,
            "total_time": final_time,
        }

    # Original path (non-gsparsity optimizers) unchanged
    required = [
        "queried_train_acc",
        "queried_val_acc",
        "scaled_queried_train_time",
        "runtime",
    ]
    if not all(key in data for key in required):
        return None

    train_acc = np.asarray(data["queried_train_acc"], dtype=float)
    valid_acc = np.asarray(data["queried_val_acc"], dtype=float)
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
        total_time = np.cumsum(eval_time)
    elif is_two_stage:
        stage2_indices = np.where(loss != -1)[0]
        if stage2_indices.size == 0:
            return None
        first_stage2_idx = stage2_indices[0]
        stage1_search_cost = runtime[:first_stage2_idx].sum()
        stage2_runtime = runtime[stage2_indices]
        stage2_cumulative_search_cost = np.cumsum(stage2_runtime)

        train_acc = train_acc[stage2_indices]
        valid_acc = valid_acc[stage2_indices]
        eval_time = eval_time[stage2_indices]

        total_time = stage1_search_cost + stage2_cumulative_search_cost + eval_time
    else:
        cumulative_search_cost = np.cumsum(runtime)
        total_time = cumulative_search_cost + eval_time

    if total_time.size == 0:
        return None

    final_time = float(total_time[-1])

    if is_random_search:
        final_train = float(np.nanmax(train_acc))
        final_valid = float(np.nanmax(valid_acc))
    else:
        final_train = float(train_acc[-1])
        final_valid = float(valid_acc[-1])

    if optimizer in ZCP_PRE_METHODS:
        extra = float(durations_map.get(dataset, 0.0))
        if np.isfinite(extra) and extra > 0:
            final_time += extra

    return {
        "train_acc": final_train,
        "valid_acc": final_valid,
        "total_time": final_time,
    }


def write_output(df: pd.DataFrame, output_path: str) -> None:
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".csv":
        df.to_csv(output_path, index=False)
    else:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="performance", index=False)
    print(f"Wrote summary with {len(df)} rows to {output_path}")


def _format_pm(mean: float, std: float, decimals: int) -> str:
    if np.isnan(mean) or np.isnan(std):
        return r"\textemdash"
    formatted = f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"
    return f"${formatted}$"


def write_latex_table(df: pd.DataFrame, output_path: str, fractional: bool) -> None:
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
            f"Final Train Acc{acc_header_suffix}",
            "final_train_acc_mean",
            "final_train_acc_std",
        ),
        (
            f"Final Val Acc{acc_header_suffix}",
            "final_valid_acc_mean",
            "final_valid_acc_std",
        ),
        ("Final Time (s)", "final_time_mean_s", "final_time_std_s"),
    ]
    col_spec = "lllccc"
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
                decimals = 4 if "Acc" in header and fractional else 2
                if "Time" in header:
                    decimals = 2
                formatted.append(_format_pm(mean, std, decimals))
        lines.append(" & ".join(formatted) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"}",
            r"\caption{For one-shot methods cumulative search cost + final evaluation; for random search cumulative evaluation cost. Final queried accuracies (mean $\pm$ std) averaged over three seeds on the NAS-Bench-201 search space.}",
            r"\label{tab:final_hp_performance}",
            r"\end{table}",
        ]
    )

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote LaTeX table to {output_path}")


# -------------------- Pairwise comparison for Final Val Acc (NEW) --------------------
ZCP_METHODS = {"jacov", "params", "synflow"}


def _is_zcp_optimizer(name: str) -> bool:
    return isinstance(name, str) and ("zcp" in name.lower())


def _cmp_scalar_rounded(
    a: float, b: float, higher_is_better: bool, decimals: int = 2
) -> int:
    """
    Round a and b, strict compare: +1 if a wins, -1 if a loses, 0 tie. NaN loses vs finite.
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
    return 1 if (ar > br if higher_is_better else ar < br) else -1


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
    Pairwise integer scores per dataset for given comparisons.
    - Round both sides to 2 decimals; strict compare; +1/-1/0.
    - Optimizer-vs-optimizer: if both sides are zcp-optimizers, compare every zcp_method variant vs every variant.
      If only one side is zcp, all its zcp variants compete vs the single non-zcp rows (many-vs-one).
    - zcp_method vs zcp_method (jacov/synflow/params): compare across the intersection of zcp-optimizers that
      have BOTH methods present on that dataset.
    - If one side empty and other has n rows: score = -n for the empty side (and +n for the other side).
    """
    if metric_col not in df.columns:
        return pd.DataFrame(columns=["comparison"] + datasets + ["sum"])
    base = df.copy()
    base[metric_col] = pd.to_numeric(base[metric_col], errors="coerce")

    out_rows = []
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
        out_rows.append(row)
    return pd.DataFrame(out_rows, columns=["comparison"] + datasets + ["sum"])


def write_pairwise_latex(df_table: pd.DataFrame, out_path: str, caption: str) -> None:
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
            cells.append(f"{int(r[ds]):+d}")
        cells.append(f"{int(r['sum']):+d}")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", r"\end{table}"]
    )
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote pairwise LaTeX table to {out_path}")


# ------------------ End pairwise comparison (NEW) --------------------


def main() -> None:
    args = parse_args()
    files = find_error_files(args.root_dir)
    if not files:
        print("No errors.json files found.", file=sys.stderr)
        return

    durations_map = load_zcp_pre_durations(args.durations_dir)

    grouped = defaultdict(list)
    for meta in files:
        stats = compute_run_stats(
            meta["path"], meta["dataset"], meta["optimizer"], durations_map
        )
        if stats is None:
            continue
        key = (
            meta["optimizer"],
            meta.get("zcp_method") or "",
            meta["dataset"],
            meta["search_space"],
        )
        grouped[key].append(stats)

    if not grouped:
        print("No valid runs found.", file=sys.stderr)
        return

    rows = []
    for (optimizer, zcp_method, dataset, search_space), metrics in sorted(
        grouped.items()
    ):
        train_vals = np.array([m["train_acc"] for m in metrics], dtype=float)
        valid_vals = np.array([m["valid_acc"] for m in metrics], dtype=float)
        time_vals = np.array([m["total_time"] for m in metrics], dtype=float)

        # Auto-scale per column: only multiply by 100 if values look fractional
        def _auto_factor(vals: np.ndarray) -> float:
            if args.fractional:
                return 1.0
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                return 1.0
            return 100.0 if np.nanmax(finite) <= 1.0 else 1.0

        factor_train = _auto_factor(train_vals)
        factor_valid = _auto_factor(valid_vals)

        rows.append(
            {
                "optimizer": optimizer,
                "dataset": dataset,
                "zcp_method": zcp_method,
                "final_train_acc_mean": float(np.mean(train_vals) * factor_train),
                "final_train_acc_std": float(np.std(train_vals, ddof=0) * factor_train),
                "final_valid_acc_mean": float(np.mean(valid_vals) * factor_valid),
                "final_valid_acc_std": float(np.std(valid_vals, ddof=0) * factor_valid),
                "final_time_mean_s": float(np.mean(time_vals)),
                "final_time_std_s": float(np.std(time_vals, ddof=0)),
            }
        )

    df = pd.DataFrame(rows)

    df = pd.DataFrame(rows)
    df.sort_values(["dataset", "optimizer", "zcp_method"], inplace=True)
    write_output(df, args.output)

    latex_output = (
        args.latex_output
        or os.path.splitext(os.path.abspath(args.output))[0] + "_table.txt"
    )
    write_latex_table(df, latex_output, args.fractional)

    # ---------------- Pairwise Final Val Acc table (NEW) ----------------
    comparisons: List[Tuple[str, str]] = [
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
    if "final_valid_acc_mean" in df.columns:
        pairwise = compute_pairwise_scores(
            df=df,
            comparisons=comparisons,
            metric_col="final_valid_acc_mean",
            datasets=ds_cols,
            higher_is_better=True,
        )
        base_root, _ = os.path.splitext(latex_output)
        out_path = f"{base_root}_pairwise_final_valid.txt"
        caption = "Pairwise scores (+1 win, -1 loss, 0 tie) on Final Mean Validation Accuracy. Values rounded to two decimals; higher is better."
        write_pairwise_latex(pairwise, out_path, caption)
    # ----------------------------


if __name__ == "__main__":
    main()
