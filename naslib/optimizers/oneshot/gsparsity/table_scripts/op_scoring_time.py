#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple, List

ZCP_METHODS = ("params", "jacov", "synflow")
OPTIMIZERS = ("zcp_gsparsity", "zcp-pre_zcp_gsparsity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute avg (runtime[1] - runtime[0]) per root path × optimizer × ZC proxy (read-only)."
    )
    p.add_argument(
        "--roots",
        type=str,
        nargs="+",
        default=[
            "naslib/optimizers/oneshot/gsparsity/result_final_hp_queried_val_acc",
            "naslib/optimizers/oneshot/gsparsity/result_final_hp",
        ],
        help="One or more root directories to scan (each containing errors.json files).",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="ImageNet16-120",
        help="Dataset to filter on (default: ImageNet16-120)",
    )
    p.add_argument(
        "--optimizers",
        type=str,
        nargs="*",
        default=list(OPTIMIZERS),
        help="Optimizers to include (default: zcp_gsparsity zcp-pre_zcp_gsparsity)",
    )
    p.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=list(ZCP_METHODS),
        help="ZC proxy methods to include (default: params jacov synflow)",
    )
    p.add_argument(
        "--latex_output",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/table_scripts/tables/runtime_delta_first_two_entries_imagenet.tex",
        help="Path to write LaTeX table.",
    )
    return p.parse_args()


def find_error_files(root_dir: str) -> Tuple[Dict, ...]:
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "errors.json" not in filenames:
            continue
        full_path = os.path.join(dirpath, "errors.json")
        relative_path = os.path.relpath(full_path, root_dir)
        parts = relative_path.split(os.sep)
        # Expected layout: optimizer/zcp_method/.../search_space/dataset/seed/errors.json
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


def load_delta_first_two_runtime(errors_json_path: str) -> Optional[float]:
    try:
        with open(errors_json_path, "r") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] Failed to read {errors_json_path}: {exc}", file=sys.stderr)
        return None
    runtime = data.get("runtime")
    if not isinstance(runtime, list) or len(runtime) < 2:
        return None
    try:
        first = float(runtime[0])
        second = float(runtime[1])
        return second - first
    except (TypeError, ValueError):
        return None


def latex_escape(text: str) -> str:
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
    out = str(text)
    for k, v in replacements.items():
        out = out.replace(k, v)
    return out


def write_latex_table(
    avg_deltas: Dict[Tuple[str, str, str, str], Dict[str, float]],
    output_path: str,
) -> None:
    # Columns: Path, Op, ZC Proxy, Dataset, Avg (s), N
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{l l l l r r}")
    lines.append(r"\toprule")
    lines.append(r"Path & Op & ZC Proxy & Dataset & Avg (s) & N \\")
    lines.append(r"\midrule")
    for key in sorted(avg_deltas.keys()):
        path_label, optimizer, method, dataset = key
        stats = avg_deltas[key]
        # Ensure rows end with '\\' (double backslash for LaTeX new line)
        lines.append(
            f"{latex_escape(path_label)} & {latex_escape(optimizer)} & {latex_escape(method)} & {latex_escape(dataset)} & {stats['avg']:.2f} & {int(stats['count'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(
        r"\caption{Average of (runtime[2nd] - runtime[1st]) per root path × optimizer × ZC proxy on ImageNet16-120 (averaged across seeds).}"
    )
    lines.append(r"\label{tab:zcp_methods_runtime_delta_imagenet}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[INFO] Wrote LaTeX table to: {output_path}")


def main() -> None:
    args = parse_args()

    # Aggregate: key = (path_label, optimizer, method, dataset) -> list of deltas across seeds
    agg: Dict[Tuple[str, str, str, str], List[float]] = {}

    for root in args.roots:
        files = find_error_files(root)
        path_label = os.path.basename(os.path.normpath(root)) or root
        for meta in files:
            if meta.get("optimizer") not in args.optimizers:
                continue
            if meta.get("dataset") != args.dataset:
                continue
            method = meta.get("zcp_method") or ""
            if method not in args.methods:
                continue
            delta = load_delta_first_two_runtime(meta["path"])
            if delta is None:
                continue
            key = (path_label, meta["optimizer"], method, meta["dataset"])
            agg.setdefault(key, []).append(delta)

    # Compute averages per group
    avg_deltas: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
    for key, deltas in agg.items():
        if not deltas:
            continue
        avg_deltas[key] = {
            "avg": sum(deltas) / float(len(deltas)),
            "count": len(deltas),
        }

    # Print summary
    print(f"[INFO] Dataset: {args.dataset}")
    print(f"[INFO] Roots: {', '.join(args.roots)}")
    for key in sorted(avg_deltas.keys()):
        path_label, optimizer, method, dataset = key
        s = avg_deltas[key]
        print(
            f"[INFO] Path={path_label}, Op={optimizer}, ZC={method}, Dataset={dataset}: N={int(s['count'])}, Avg={s['avg']:.2f}s"
        )

    # Write LaTeX table (Path, Op, ZC Proxy, Dataset, Avg, N)
    write_latex_table(avg_deltas=avg_deltas, output_path=args.latex_output)


if __name__ == "__main__":
    main()
