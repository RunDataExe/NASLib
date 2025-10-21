import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

DATASET_CLASSES = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}


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
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp/final_performance_summary.xlsx",
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


def compute_run_stats(filepath: str, dataset: str) -> Optional[Dict[str, float]]:
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
    if not all(key in data for key in required):
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

    return {
        "train_acc": float(train_acc[-1]),
        "valid_acc": float(valid_acc[-1]),
        "total_time": float(total_time[-1]),
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


def main() -> None:
    args = parse_args()
    files = find_error_files(args.root_dir)
    if not files:
        print("No errors.json files found.", file=sys.stderr)
        return

    grouped = defaultdict(list)
    for meta in files:
        stats = compute_run_stats(meta["path"], meta["dataset"])
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

        factor = 1.0 if args.fractional else 100.0
        rows.append(
            {
                "optimizer": optimizer,
                "dataset": dataset,
                "zcp_method": zcp_method,
                "final_train_acc_mean": float(np.mean(train_vals) * factor),
                "final_train_acc_std": float(np.std(train_vals, ddof=0) * factor),
                "final_valid_acc_mean": float(np.mean(valid_vals) * factor),
                "final_valid_acc_std": float(np.std(valid_vals, ddof=0) * factor),
                "final_time_mean_s": float(np.mean(time_vals)),
                "final_time_std_s": float(np.std(time_vals, ddof=0)),
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(["dataset", "optimizer", "zcp_method"], inplace=True)
    write_output(df, args.output)

    latex_output = (
        args.latex_output
        or os.path.splitext(os.path.abspath(args.output))[0] + "_table.txt"
    )
    write_latex_table(df, latex_output, args.fractional)


if __name__ == "__main__":
    main()
