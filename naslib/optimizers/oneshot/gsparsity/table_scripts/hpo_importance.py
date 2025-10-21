import os
import argparse
import optuna
import pandas as pd
from optuna.importance import FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator

# Known ZCP tags
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

def parse_study_identity(study_name: str):
    """
    Parse study_name to (optimizer, search_space, dataset, seed, zcp_method)
    Accepts names like:
      optimizer-<...>-nasbench201-<dataset tokens>-<seed>[-<zcp>]
    """
    parts = study_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected study name format: {study_name}")

    # strip trailing tokens until reaching a seed or a known zcp token
    tmp = list(parts)
    while tmp:
        tail = tmp[-1]
        if tail in ZCP_METHODS:
            break
        try:
            int(tail)
            break
        except ValueError:
            tmp.pop()
    if len(tmp) < 4:
        raise ValueError(f"Unexpected study name after suffix stripping: {study_name}")

    zcp_method = None
    if tmp[-1] in ZCP_METHODS:
        zcp_method = tmp.pop()
    seed_token = tmp.pop()
    try:
        study_seed = int(seed_token)
    except ValueError as exc:
        raise ValueError(f"Seed is not an int in {study_name}") from exc

    search_spaces = {"nasbench201", "nasbench301"}
    try:
        search_space_idx = max(i for i, token in enumerate(tmp) if token in search_spaces)
    except ValueError as exc:
        raise ValueError(f"Search space token not found in {study_name}") from exc

    search_space = tmp[search_space_idx]
    optimizer = "-".join(tmp[:search_space_idx]).strip("-")
    dataset = "-".join(tmp[search_space_idx + 1 :]).strip("-")
    if not optimizer or not dataset:
        raise ValueError(f"Could not parse optimizer/dataset from {study_name}")
    return optimizer, search_space, dataset, study_seed, zcp_method

def parse_args():
    p = argparse.ArgumentParser(description="Generate combined hyperparameter importance table (all methods).")
    p.add_argument(
        "--db_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/results_wide_hpo/WHPO_Databases",
        help="Directory containing Optuna study databases or a single .db file."
    )
    p.add_argument(
        "--output",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/table_scripts/tables/hyperparameter_importance_tables.xlsx",
        help="Output Excel file (sheet: all_methods_importance)."
    )
    p.add_argument(
        "--importance-method",
        choices=["fanova", "mdi"],
        default="fanova",
        help="Method to compute importance."
    )
    p.add_argument(
        "--latex-output",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/table_scripts/tables/hpo_importance_all_methods_table.txt",
        help="Optional LaTeX output path (.txt). Defaults to <output> base + _all_methods_importance_table.txt"
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places for numeric values in LaTeX output."
    )
    return p.parse_args()

def list_db_files(db_dir):
    if os.path.isdir(db_dir):
        return [os.path.join(db_dir, f) for f in os.listdir(db_dir) if f.endswith(".db")]
    if os.path.isfile(db_dir) and db_dir.endswith(".db"):
        return [db_dir]
    raise FileNotFoundError(f"db_dir must be a directory or .db file: {db_dir}")

def compute_importances(study, method):
    if method == "fanova":
        evaluator = FanovaImportanceEvaluator()
    else:
        evaluator = MeanDecreaseImpurityImportanceEvaluator()
    return evaluator.evaluate(study)

def extract_all_rows(db_files, importance_method):
    """
    Build a single list of rows, each row contains:
      - optimizer, dataset, zcp_method, study_seed, study_name
      - one key per hyperparameter with importance value
    """
    rows = []
    for db_path in db_files:
        storage = f"sqlite:///{os.path.abspath(db_path)}"
        try:
            summaries = optuna.study.get_all_study_summaries(storage)
        except Exception as e:
            print(f"Error listing studies from {db_path}: {e}")
            continue

        for s in summaries:
            study_name = s.study_name
            try:
                optimizer, search_space, dataset, study_seed, zcp_method = parse_study_identity(study_name)
            except Exception as e:
                print(f"[WARN] Skipping study {study_name}: {e}")
                continue

            try:
                study = optuna.load_study(storage=storage, study_name=study_name)
                importances = compute_importances(study, importance_method)
            except Exception as e:
                print(f"Error processing study {study_name}: {e}")
                continue

            row = {
                "optimizer": optimizer,
                "dataset": dataset,
                "zcp_method": zcp_method or "",
                "study_seed": study_seed,
                "study_name": study_name,
            }
            for p, imp in importances.items():
                row[p] = imp
            rows.append(row)
    return rows

def save_combined_excel(rows, output_file):
    if not rows:
        print("No rows to save.")
        return
    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["dataset", "optimizer", "zcp_method", "study_seed"] if c in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_methods_importance", index=False)

# ---- LaTeX helpers ----
def _latex_escape(text):
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s

def _format_numeric(value, decimals=4):
    if value is None:
        return "—"
    try:
        f = float(value)
    except Exception:
        return _latex_escape(str(value))
    return f"{f:.{decimals}f}"

def build_latex_all_methods(rows, title, label, decimals=4):
    """
    Build one LaTeX table:
      Method | Dataset | ZC Proxy | <param1> | <param2> | ...
    Cell values are importances. If a hyperparameter is absent for a method, print —.
    """
    if not rows:
        return ""

    exclude_cols = {"optimizer", "dataset", "zcp_method", "study_seed", "study_name"}
    param_names = sorted({k for r in rows for k in r.keys() if k not in exclude_cols})

    col_spec = "lll" + "r" * len(param_names)
    header = " & ".join(
        [
            r"\textbf{Method}",
            r"\textbf{Dataset}",
            r"\textbf{ZC Proxy}",
            *[f"\\textbf{{{_latex_escape(p)}}}" for p in param_names],
        ]
    )

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\rotatebox{90}{",
        r"\begin{adjustbox}{max width=\textheight}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]

    sorted_rows = sorted(rows, key=lambda r: (r.get("dataset",""), r.get("optimizer",""), r.get("zcp_method","")))
    for r in sorted_rows:
        method = _latex_escape(r.get("optimizer", ""))
        dataset = _latex_escape(r.get("dataset", ""))
        zcp = _latex_escape(r.get("zcp_method") or "-")
        vals = [_format_numeric(r.get(p, None), decimals=decimals) for p in param_names]
        lines.append(" & ".join([method, dataset, zcp] + vals) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"}",
            rf"\caption{{{_latex_escape(title)}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)

def main():
    args = parse_args()
    db_files = list_db_files(args.db_dir)
    rows = extract_all_rows(db_files, args.importance_method)

    # Excel (combined)
    save_combined_excel(rows, args.output)

    # LaTeX output path
    latex_output = args.latex_output.strip()
    if not latex_output:
        base = os.path.splitext(os.path.abspath(args.output))[0]
        latex_output = base + "_all_methods_importance_table.txt"

    latex = build_latex_all_methods(
        rows,
        title=f"Hyperparameter importance ({args.importance_method}) across all methods.",
        label="tab:hp_importance_all_methods",
        decimals=args.decimals,
    )
    os.makedirs(os.path.dirname(os.path.abspath(latex_output)), exist_ok=True)
    with open(latex_output, "w") as fh:
        fh.write(latex)
    print(f"Wrote LaTeX table to {latex_output}")
    print(f"Saved Excel table to {args.output}")

if __name__ == "__main__":
    main()