import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import json
import optuna
import pandas as pd
from optuna.importance import (
    FanovaImportanceEvaluator,
    MeanDecreaseImpurityImportanceEvaluator,
)
from optuna.trial import FrozenTrial, TrialState

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export best Optuna hyperparameters and importances to Excel."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="SQLite DB file (*.db) or directory containing Optuna studies.",
    )
    parser.add_argument(
        "--output",
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp/final_hp_summary.xlsx",
        help="Excel file to create.",
    )
    parser.add_argument(
        "--importance-method",
        choices=["fanova", "mdi", "none"],
        default="fanova",
        help="Optuna importance evaluator to use.",
    )
    return parser.parse_args()


def list_db_files(source: str) -> List[str]:
    if os.path.isdir(source):
        return sorted(
            os.path.join(source, f) for f in os.listdir(source) if f.endswith(".db")
        )
    if os.path.isfile(source) and source.endswith(".db"):
        return [source]
    raise FileNotFoundError(f"Source must be a .db file or directory: {source}")


def list_studies(db_path: str) -> List[optuna.study.StudySummary]:
    storage = f"sqlite:///{os.path.abspath(db_path)}"
    try:
        return optuna.study.get_all_study_summaries(storage)
    except Exception as exc:
        print(f"[WARN] Cannot list studies in {db_path}: {exc}", file=sys.stderr)
        return []


def open_study(db_path: str, study_name: str) -> optuna.study.Study:
    storage = f"sqlite:///{os.path.abspath(db_path)}"
    return optuna.load_study(storage=storage, study_name=study_name)


def _trial_budget(trial: FrozenTrial) -> int:
    budget = trial.user_attrs.get("budget")
    if isinstance(budget, (int, float)):
        return int(budget)
    if trial.last_step is not None:
        return int(trial.last_step)
    if trial.intermediate_values:
        return int(max(trial.intermediate_values.keys()))
    return 0


def _trial_score_highest_val_acc(trial: FrozenTrial) -> Optional[float]:
    if trial.state == TrialState.COMPLETE:
        return trial.value
    if trial.state == TrialState.PRUNED:
        if trial.last_step is not None and trial.last_step in trial.intermediate_values:
            return trial.intermediate_values[trial.last_step]
        if trial.intermediate_values:
            last_step = max(trial.intermediate_values.keys())
            return trial.intermediate_values[last_step]
    return None


def _eligible_trials(study: optuna.study.Study) -> List[FrozenTrial]:
    trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE, TrialState.PRUNED]
    )
    return [t for t in trials if not t.user_attrs.get("internal_early_stopped", False)]


def select_best_trial_highest_val_acc(study: optuna.study.Study) -> FrozenTrial:
    best_trial: Optional[FrozenTrial] = None
    best_score: Optional[float] = None
    for trial in _eligible_trials(study):
        score = _trial_score_highest_val_acc(trial)
        if score is not None and (best_score is None or score > best_score):
            best_score = score
            best_trial = trial
    if best_trial is None:
        raise ValueError("No eligible trials with comparable scores.")
    return best_trial


class _FilteredStudyProxy:
    def __init__(self, study: optuna.study.Study, trials: List[FrozenTrial]):
        self._study = study
        self._trials = list(trials)

    def get_trials(self, deepcopy: bool = True, states=None):
        trials = self._trials
        if states is not None:
            states_set = set(states)
            trials = [t for t in trials if t.state in states_set]
        if deepcopy:
            import copy

            return [copy.deepcopy(t) for t in trials]
        return trials

    @property
    def directions(self):
        return self._study.directions

    @property
    def direction(self):
        return self._study.direction

    def __getattr__(self, item):
        return getattr(self._study, item)


def compute_importances(
    study: optuna.study.Study,
    eligible_trials: List[FrozenTrial],
    params: List[str],
    method: str,
) -> Dict[str, float]:
    if method == "none" or not params:
        return {}
    evaluator = (
        FanovaImportanceEvaluator(seed=0)
        if method == "fanova"
        else MeanDecreaseImpurityImportanceEvaluator()
    )
    proxy = _FilteredStudyProxy(study, eligible_trials)
    try:
        return evaluator.evaluate(proxy, params=params)
    except Exception as exc:
        print(
            f"[WARN] Importance calculation failed for {study.study_name}: {exc}",
            file=sys.stderr,
        )
        return {}


def parse_study_identity(study_name: str) -> Tuple[str, str, str, int, Optional[str]]:
    parts = study_name.split("-")
    if len(parts) < 4:
        raise ValueError(f"Unexpected study name format: {study_name}")
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


def latex_escape(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text)
    stripped = text.strip()
    if stripped.startswith("$") and stripped.endswith("$"):
        return text
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
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def stringify_param_value(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "None"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def latexify_pm(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text)
    if "±" in s:
        left, right = s.split("±", 1)
        return f"${left.strip()} \\pm {right.strip()}$"
    if r"\pm" in s:
        stripped = s.strip()
        if stripped.startswith("$") and stripped.endswith("$"):
            return stripped
        return f"${stripped}$"
    return s


def format_numeric(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_latex_table(best_rows: List[Dict[str, object]]) -> str:
    if not best_rows:
        return ""
    param_names = sorted(
        {
            key.split("param.", 1)[1]
            for row in best_rows
            for key in row.keys()
            if key.startswith("param.")
        }
    )
    # Add an extra numeric column for the trials count
    col_spec = "lllrrrr" + "l" * len(param_names)
    header_cols = [
        r"\textbf{Method}",
        r"\textbf{Dataset}",
        r"\textbf{ZC Proxy}",
        r"\textbf{Trial \#}",
        r"\textbf{Trial Value}",
        r"\textbf{Trial Budget}",
        r"\textbf{Trials}",
        *[f"\\textbf{{{latex_escape(name)}}}" for name in param_names],
    ]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\rotatebox{90}{",
        r"\begin{adjustbox}{max width=\textheight}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]
    sorted_rows = sorted(
        best_rows,
        key=lambda r: (
            r.get("dataset", ""),
            r.get("optimizer", ""),
            r.get("zcp_method", ""),
        ),
    )
    for row in sorted_rows:
        optimizer = latex_escape(row.get("optimizer", ""))
        dataset = latex_escape(row.get("dataset", ""))
        zcp = latex_escape(row.get("zcp_method") or "None")
        trial_no = latex_escape(str(row.get("selected_trial", "—")))
        trial_value_raw = row.get("trial_value")
        if trial_value_raw is None:
            trial_value_raw = row.get("trial_score")
        trial_value = latex_escape(latexify_pm(format_numeric(trial_value_raw)))
        trial_budget = latex_escape(
            latexify_pm(format_numeric(row.get("trial_budget")))
        )
        # Prefer eligible trials, fall back to total if missing
        trials_count = row.get("n_trials_eligible", row.get("n_trials_total"))
        trials_count = latex_escape(format_numeric(trials_count))

        param_values = []
        for name in param_names:
            key = f"param.{name}"
            if key in row:
                formatted_value = stringify_param_value(row.get(key))
                formatted_value = latex_escape(latexify_pm(formatted_value))
            else:
                formatted_value = latex_escape("—")
            param_values.append(formatted_value)
        line = " & ".join(
            [
                optimizer,
                dataset,
                zcp,
                trial_no,
                trial_value,
                trial_budget,
                trials_count,
                *param_values,
            ]
        )
        lines.append(line + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{adjustbox}",
            r"}",
            r"\caption{Final Optuna-selected hyperparameters per method, dataset, and zero-cost proxy.}",
            r"\label{tab:final_hp_settings}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    db_files = list_db_files(args.source)

    best_rows: List[Dict[str, object]] = []
    importance_rows: List[Dict[str, object]] = []

    for db_path in db_files:
        summaries = list_studies(db_path)
        if not summaries:
            continue
        for summary in summaries:
            study_name = summary.study_name
            try:
                optimizer, search_space, dataset, study_seed, zcp_method = (
                    parse_study_identity(study_name)
                )
            except Exception as exc:
                print(f"[WARN] Skipping study {study_name}: {exc}", file=sys.stderr)
                continue
            try:
                study = open_study(db_path, study_name)
                eligible_trials = _eligible_trials(study)
                if not eligible_trials:
                    raise ValueError("No eligible trials after filtering.")
                best_trial = select_best_trial_highest_val_acc(study)
            except Exception as exc:
                print(
                    f"[WARN] Skipping study {study_name} (selection failed): {exc}",
                    file=sys.stderr,
                )
                continue

            base_info = {
                "db_path": db_path,
                "study_name": study_name,
                "optimizer": optimizer,
                "search_space": search_space,
                "dataset": dataset,
                "study_seed": study_seed,
                "zcp_method": zcp_method or "",
                "selected_trial": best_trial.number,
                "trial_state": best_trial.state.name,
                "trial_value": best_trial.value,
                "trial_score": _trial_score_highest_val_acc(best_trial),
                "trial_budget": _trial_budget(best_trial),
                "trial_last_step": best_trial.last_step,
                # New: counts for reporting
                "n_trials_total": len(study.get_trials(deepcopy=False)),
                "n_trials_eligible": len(eligible_trials),
            }

            row = dict(base_info)
            for param_name, param_value in best_trial.params.items():
                row[f"param.{param_name}"] = param_value
            best_rows.append(row)

            importances = compute_importances(
                study=study,
                eligible_trials=eligible_trials,
                params=sorted(best_trial.params.keys()),
                method=args.importance_method,
            )
            for param_name, importance in importances.items():
                importance_rows.append(
                    {
                        **base_info,
                        "param": param_name,
                        "importance": importance,
                    }
                )

    if not best_rows:
        print("No eligible trials found; no Excel file written.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    best_df = pd.DataFrame(best_rows)
    importance_df = pd.DataFrame(importance_rows)

    if not best_df.empty:
        best_df.sort_values(
            ["dataset", "optimizer", "zcp_method", "study_seed"], inplace=True
        )
    if not importance_df.empty:
        importance_df.sort_values(
            ["dataset", "optimizer", "zcp_method", "study_seed", "param"],
            inplace=True,
        )

    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        best_df.to_excel(writer, sheet_name="best_params", index=False)
        if not importance_df.empty:
            importance_df.to_excel(writer, sheet_name="param_importance", index=False)

    latex_table = build_latex_table(best_rows)
    if latex_table:
        table_path = os.path.splitext(os.path.abspath(args.output))[0] + "_table.txt"
        with open(table_path, "w") as fh:
            fh.write(latex_table)
        print(f"Wrote LaTeX table to {table_path}")

    print(f"Exported {len(best_rows)} study rows to {args.output}")


if __name__ == "__main__":
    main()
