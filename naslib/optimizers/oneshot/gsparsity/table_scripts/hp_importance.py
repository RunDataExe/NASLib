import os
import argparse
import optuna
import pandas as pd
from optuna.importance import FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator

# New: common ZCP set and study-name parsing to match final_hp_setting_table.py
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
    to align with final_hp_setting_table.py.
    """
    parts = study_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected study name format: {study_name}")
    # Strip trailing tokens until reaching a seed or a known zcp token
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
    seed_token = parts.pop()
    try:
        study_seed = int(seed_token)
    except ValueError as exc:
        raise ValueError(f"Seed is not an int in {study_name}") from exc
    search_spaces = {"nasbench201", "nasbench301"}
    try:
        search_space_idx = max(
            i for i, token in enumerate(parts) if token in search_spaces
        )
    except ValueError as exc:
        raise ValueError(f"Search space token not found in {study_name}") from exc
    search_space = parts[search_space_idx]
    optimizer = "-".join(parts[:search_space_idx]).strip("-")
    dataset = "-".join(parts[search_space_idx + 1 :]).strip("-")
    if not optimizer or not dataset:
        raise ValueError(f"Could not parse optimizer/dataset from {study_name}")
    return optimizer, search_space, dataset, study_seed, zcp_method


def parse_args():
    parser = argparse.ArgumentParser(description="Generate hyperparameter importance tables.")
    parser.add_argument(
        "--db_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/results_wide_hpo/WHPO_Databases",
        help="Directory containing Optuna study databases."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hyperparameter_importance_tables.xlsx",
        help="Output Excel file for the importance tables."
    )
    parser.add_argument(
        "--importance-method",
        choices=["fanova", "mdi"],
        default="fanova",
        help="Method to compute importance."
    )
    parser.add_argument(
        "--latex-output",
        type=str,
        default="",
        help="Optional LaTeX output path (.txt). Defaults to <output> base + _importance_table.txt"
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places for numeric values in LaTeX output."
    )
    return parser.parse_args()

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

def extract_importance_data(db_files, importance_method):
    """
    Returns two lists of rows (dict):
      - gsparsity_importance_rows
      - darts_importance_rows
    Each row has keys: optimizer, dataset, zcp_method, then one key per hyperparameter with its importance.
    """
    gsparsity_rows = []
    darts_rows = []

    for db_path in db_files:
        storage = f"sqlite:///{os.path.abspath(db_path)}"
        try:
            studies = optuna.study.get_all_study_summaries(storage)
        except Exception as e:
            print(f"Error listing studies from {db_path}: {e}")
            continue

        for study_summary in studies:
            study_name = study_summary.study_name
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
                # Optionally keep seed if needed for debugging/Excel (not used in LaTeX columns)
                "study_seed": study_seed,
                "study_name": study_name,
            }
            # Merge the importance values
            for p, imp in importances.items():
                row[p] = imp

            # Classify into gsparsity/darts groups based on optimizer token
            opt_lower = optimizer.lower()
            if "gsparsity" in opt_lower:
                gsparsity_rows.append(row)
            elif "darts" in opt_lower:
                darts_rows.append(row)
            else:
                # Fallback: keep in gsparsity group by default
                gsparsity_rows.append(row)

    return gsparsity_rows, darts_rows

def save_to_excel(gsparsity_rows, darts_rows, output_file):
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        if gsparsity_rows:
            gsparsity_df = pd.DataFrame(gsparsity_rows)
            # Sort for readability
            sort_cols = [c for c in ["dataset", "optimizer", "zcp_method", "study_seed"] if c in gsparsity_df.columns]
            if sort_cols:
                gsparsity_df.sort_values(sort_cols, inplace=True)
            gsparsity_df.to_excel(writer, sheet_name="gsparsity_importance", index=False)
        if darts_rows:
            darts_df = pd.DataFrame(darts_rows)
            sort_cols = [c for c in ["dataset", "optimizer", "zcp_method", "study_seed"] if c in darts_df.columns]
            if sort_cols:
                darts_df.sort_values(sort_cols, inplace=True)
            darts_df.to_excel(writer, sheet_name="darts_importance", index=False)

# --- New: LaTeX helpers and table writer styled like final_hp_setting_table.py ---
def _latex_escape(text):
    if text is None:
        return ""
    s = str(text)
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

def _build_latex_importance_table(rows, title, label, decimals=4):
    """
    Build a LaTeX table with columns:
      Method | Dataset | ZC Proxy | <param1> | <param2> | ...
    Values are the importance per param for that (Method, Dataset, ZC Proxy) row.
    Param columns are the union across rows, sorted for a consistent order.
    """
    if not rows:
        return ""

    # Collect union of parameter names (exclude metadata columns)
    exclude_cols = {"optimizer", "dataset", "zcp_method", "study_seed", "study_name"}
    param_names = sorted({k for row in rows for k in row.keys() if k not in exclude_cols})

    # Column spec: 3 left-aligned id columns, then right-aligned numeric columns
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

    # Sort rows for a stable order (dataset, optimizer, zcp)
    sorted_rows = sorted(
        rows, key=lambda r: (r.get("dataset", ""), r.get("optimizer", ""), r.get("zcp_method", ""))
    )

    for row in sorted_rows:
        method = _latex_escape(row.get("optimizer", ""))
        dataset = _latex_escape(row.get("dataset", ""))
        zcp = _latex_escape(row.get("zcp_method") or "-")
        vals = [_format_numeric(row.get(p, None), decimals=decimals) for p in param_names]
        lines.append(" & ".join([method, dataset, zcp] + vals) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"}",
            rf"\caption{{{_latex_escape(title)}}}",
            rf"\label{{{_latex_escape(label)}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)

def save_latex_tables(gsparsity_rows, darts_rows, importance_method, latex_output_base, decimals=4):
    tables = []
    if gsparsity_rows:
        title = f"Hyperparameter importance ({importance_method}) for gsparsity studies."
        tables.append(_build_latex_importance_table(gsparsity_rows, title, "tab:hp_importance_gsparsity", decimals=decimals))
    if darts_rows:
        title = f"Hyperparameter importance ({importance_method}) for DARTS studies."
        tables.append(_build_latex_importance_table(darts_rows, title, "tab:hp_importance_darts", decimals=decimals))

    if not tables:
        return None

    output_path = latex_output_base
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("\n\n".join(tables))
    return output_path
# --- End new LaTeX ---

def main():
    args = parse_args()
    db_files = list_db_files(args.db_dir)

    gsparsity_rows, darts_rows = extract_importance_data(db_files, args.importance_method)

    save_to_excel(gsparsity_rows, darts_rows, args.output)

    # Determine LaTeX output path
    latex_output = args.latex_output.strip()
    if not latex_output:
        base = os.path.splitext(os.path.abspath(args.output))[0]
        latex_output = base + "_importance_table.txt"

    latex_path = save_latex_tables(
        gsparsity_rows, darts_rows, args.importance_method, latex_output, decimals=args.decimals
    )
    if latex_path:
        print(f"Wrote LaTeX table(s) to {latex_path}")

    print(f"Saved hyperparameter importance tables to {args.output}")

if __name__ == "__main__":
    main()