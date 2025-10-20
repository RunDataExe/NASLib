import os, glob, argparse, json
from collections import defaultdict
import numpy as np
import pandas as pd
import optuna

# Matplotlib to match your house style
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# add: known search spaces from NASLib
from naslib.search_spaces import supported_search_spaces

# ---- Style copied to align with your other scripts ----
DEFAULTS = [
    (0.5490196078431373, 0.19215686274509805, 1.0),
    (0.20392156862745098, 0.8901960784313725, 0.3411764705882353),
    (0.9764705882352941, 0.0, 0.6313725490196078),
    (0.1803921568627451, 0.32941176470588235, 0.0),
    (0.00392156862745098, 0.29411764705882354, 0.6666666666666666),
    (0.7843137254901961, 0.8, 0.44313725490196076),
    (1.0, 0.5137254901960784, 0.34509803921568627),
    (0.00392156862745098, 0.5686274509803921, 0.5254901960784314),
    (0.5568627450980392, 0.43137254901960786, 0.0),
    (0.9725490196078431, 0.7254901960784313, 0.5607843137254902),
]
C_MAX = 10
COLORS = [*DEFAULTS[:C_MAX]] * 3
FMTS = [*["-"] * C_MAX, *["--"] * C_MAX, *[":"] * C_MAX]
MARKERS = ["o", "s", "+", "D", "x", "^", "*", "v", "<", ">", "p", "h", "H", "P"]

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"

def _sort_key_method(m):
    # m is (optimizer, zcp_method); map None/NaN to '' for stable sorting
    return (m[0] or "", "" if (m[1] is None or (isinstance(m[1], float) and np.isnan(m[1]))) else m[1])

# ---- robust filename parsing ----
def _is_int_token(tok: str) -> bool:
    try:
        int(tok)
        return True
    except Exception:
        return False

def canonical_optimizer_name(opt: str) -> str:
    """
    Canonicalize optimizer names for merging with final runs.
    Do not change labels used for plotting; only use this for joins.
    """
    if "zcp_gsparsity" in opt:
        return "zcp_gsparsity"
    if "gsparsity" in opt and "zcp" not in opt:
        return "gsparsity"
    if "darts" in opt:
        return "darts"
    return opt

def parse_db_basename(name: str) -> dict:
    """
    Parse basename without .db:
      [optimizer]-[search_space]-[dataset...]-[seed][-zcp_method]
    optimizer and dataset may contain '-' (e.g., zcp-pre_gsparsity, ImageNet16-120).
    """
    tokens = name.split("-")

    # 1) last token: either zcp_method or seed
    if _is_int_token(tokens[-1]):
        zcp_method = None
        seed_tok_idx = len(tokens) - 1
    else:
        zcp_method = tokens[-1]
        seed_tok_idx = len(tokens) - 2

    if seed_tok_idx < 0 or not _is_int_token(tokens[seed_tok_idx]):
        raise ValueError(f"Cannot parse seed from study name: {name}")
    seed = tokens[seed_tok_idx]

    # 2) locate search space from known set
    ss_names = set(supported_search_spaces.keys())
    ss_idx = None
    for i, t in enumerate(tokens[:seed_tok_idx]):
        if t in ss_names:
            ss_idx = i
            break
    if ss_idx is None:
        raise ValueError(f"Cannot find search_space in: {name}")

    search_space = tokens[ss_idx]
    optimizer = "-".join(tokens[:ss_idx])
    dataset = "-".join(tokens[ss_idx + 1 : seed_tok_idx])

    if not optimizer or not dataset:
        raise ValueError(f"Cannot parse optimizer/dataset in: {name}")

    return {
        "optimizer": optimizer,
        "search_space": search_space,
        "dataset": dataset,
        "seed": seed,
        "zcp_method": zcp_method,
        "study_name": name,
    }

# ---- Helpers ----
def parse_meta(db_path):
    name = os.path.basename(db_path).replace(".db", "")
    return parse_db_basename(name)

def method_key(meta):
    # keep zcp method in the key so each variant is a distinct method
    return (meta["optimizer"], meta["zcp_method"])

def format_method_label(opt, zcp):
    # Avoid "(nan)" and empty tags
    if zcp is None:
        return f"{opt}"
    if isinstance(zcp, float) and np.isnan(zcp):
        return f"{opt}"
    z = str(zcp).strip()
    return f"{opt}" if z.lower() == "nan" or z == "" else f"{opt} ({z})"

# --- shared style maps so both plots use the same assignment ---
def build_style_maps(metas):
    method_list = sorted({(m["optimizer"], m["zcp_method"]) for m in metas}, key=_sort_key_method)
    method_to_color  = {m: COLORS[i % len(COLORS)]   for i, m in enumerate(method_list)}
    method_to_fmt    = {m: FMTS[i % len(FMTS)]       for i, m in enumerate(method_list)}
    method_to_marker = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(method_list)}
    return method_list, method_to_color, method_to_fmt, method_to_marker

def trial_best_value(trial: optuna.trial.FrozenTrial):
    # use final value if COMPLETE; last intermediate for PRUNED/FAIL if available
    if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
        return float(trial.value)
    if trial.intermediate_values:
        last_k = max(trial.intermediate_values.keys())
        v = trial.intermediate_values.get(last_k)
        if v is not None:
            return float(v)
    # fallback: None
    return None

def best_over_time(trials):
    # Staircase incumbent vs wall-clock minutes (trial completion)
    starts = [t.datetime_start for t in trials if t.datetime_start]
    if not starts:
        return np.array([]), np.array([])
    t0 = min(starts)
    events = []
    for t in trials:
        v = trial_best_value(t)
        if v is None or t.datetime_complete is None:
            continue
        x = (t.datetime_complete - t0).total_seconds() / 60.0
        events.append((x, v))
    if not events:
        return np.array([]), np.array([])
    events.sort()
    xs, ys = [], []
    best = -np.inf
    for x, v in events:
        best = max(best, v)
        xs.append(x); ys.append(best)
    return np.array(xs), np.array(ys)

def incumbent_at(xs, ys, T):
    if len(xs) == 0:
        return np.nan
    idx = np.searchsorted(xs, T, side="right") - 1
    return ys[idx] if idx >= 0 else np.nan

def normalized_auc(xs, ys, T, ref=None, y0: float = 0.0):
    """
    AUC of the incumbent curve on [0, T] using a right-continuous staircase.
    - xs, ys: incumbent at trial completion times (ys non-decreasing)
    - ref: if provided, normalize accuracy by this common reference
    - y0: incumbent at t=0 (use 0.0 to avoid counting area before first trial)
    Returns AUC/T.
    """
    if len(xs) == 0 or T <= 0:
        return np.nan
    mask = xs <= T
    X = np.concatenate([[0.0], xs[mask], [T]])
    last_y = ys[np.searchsorted(xs, T, side="right") - 1] if np.any(mask) else y0
    Y = np.concatenate([[y0], ys[mask], [last_y]])
    if ref is not None and np.isfinite(ref) and ref > 0:
        Y = np.clip(Y / ref, 0.0, 1.0)
    return float(np.trapz(Y, X) / T)

def load_studies(db_dir, search_space_filter=None):
    dbs = sorted(glob.glob(os.path.join(db_dir, "*.db")))
    metas, studies = [], []
    for db in dbs:
        try:
            meta = parse_meta(db)
        except Exception as e:
            print(f"Skip {db}: cannot parse filename ({e})")
            continue
        if search_space_filter and meta["search_space"] != search_space_filter:
            continue
        try:
            st = optuna.load_study(storage=f"sqlite:///{db}", study_name=meta["study_name"])
        except Exception as e:
            print(f"Skip {db}: {e}")
            continue
        metas.append(meta); studies.append(st)
    return metas, studies

# ---- Final runs loader (errors.json), no CSV needed ----
def find_error_files(root_dir):
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "errors.json" in filenames:
            p = os.path.join(dirpath, "errors.json")
            rel = os.path.relpath(p, root_dir)
            parts = rel.split(os.sep)
            # With zcp: {optimizer}/{zcp}/{ss}/{dataset}/{seed}/errors.json
            # Without:  {optimizer}/{ss}/{dataset}/{seed}/errors.json
            if len(parts) >= 5:
                optimizer = parts[0]
                seed = parts[-2]
                dataset = parts[-3]
                search_space = parts[-4]
                zcp_method = parts[1] if len(parts) >= 6 else None
                files.append(dict(path=p, optimizer=optimizer, zcp_method=zcp_method,
                                  dataset=dataset, search_space=search_space, seed=seed))
    return files

def load_final_runs(final_root_dir):
    if not final_root_dir or not os.path.isdir(final_root_dir):
        return None
    files = find_error_files(final_root_dir)
    if not files:
        return None
    rows = []
    for f in files:
        try:
            with open(f["path"], "r") as fh:
                data = json.load(fh)
            qva = data.get("queried_val_acc", None)
            if not isinstance(qva, list) or len(qva) == 0:
                continue
            rows.append({
                "dataset": f["dataset"],
                "optimizer": f["optimizer"],
                "zcp_method": f["zcp_method"],  # may be None
                "seed": f["seed"],
                "final_acc": float(qva[-1]),
            })
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # keep None groups (dropna=False)
    agg = df.groupby(["dataset","optimizer","zcp_method"], dropna=False, as_index=False)["final_acc"] \
            .agg(mean_final="mean", std_final="std", seeds="count")
    return agg

# ---- Reports ----
def make_fixed_time_reports(metas, studies, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Build method style mapping (consistent order across plots)
    method_list, method_to_color, method_to_fmt, method_to_marker = build_style_maps(metas)

    # Build curves grouped by dataset/method
    per = defaultdict(list)
    for meta, st in zip(metas, studies):
        xs, ys = best_over_time(st.trials)
        if len(xs) == 0:
            continue
        per[(meta["dataset"], method_key(meta))].append(dict(xs=xs, ys=ys, meta=meta))

    # Decide common horizons per dataset
    horizons = {}
    for dataset in {m["dataset"] for m in metas}:
        last_times = []
        for (ds, mkey), runs in per.items():
            if ds != dataset:
                continue
            for r in runs:
                last_times.append(r["xs"][-1])
        if not last_times:
            continue
        T2 = float(np.min(last_times))
        T1 = max(1e-6, 0.1 * T2)
        horizons[dataset] = (T1, T2)

    # Dataset-level accuracy reference (optional; used only for auc_norm_T2)
    dataset_ref = {}
    for ds in horizons.keys():
        bests = []
        for (dsm, _), runs in per.items():
            if dsm != ds:
                continue
            for r in runs:
                if len(r["ys"]) > 0:
                    bests.append(np.nanmax(r["ys"]))
        dataset_ref[ds] = float(np.nanmax(bests)) if bests else np.nan

    # Tables and plots
    rows, auc_rows = [], []
    for dataset in sorted({m["dataset"] for m in metas}):
        if dataset not in horizons:
            continue
        T1, T2 = horizons[dataset]
        ref_ds = dataset_ref.get(dataset, np.nan)

        fig = plt.figure(figsize=(14, 8))
        ax = plt.gca()
        # collect legend texts with AUC per method (for this dataset)
        legend_texts = {}

        for mkey in method_list:
            runs = per.get((dataset, mkey), [])
            if not runs:
                continue
            color = method_to_color[mkey]
            fmt   = method_to_fmt[mkey]
            marker= method_to_marker[mkey]
            label = format_method_label(mkey[0], mkey[1])

            xs = runs[0]["xs"]; ys = runs[0]["ys"]

            # Metrics within dataset
            acc_T1   = incumbent_at(xs, ys, T1)
            acc_T2   = incumbent_at(xs, ys, T2)
            auc_raw  = normalized_auc(xs, ys, T2, ref=None,  y0=0.0)     # use this for within-dataset comparison
            auc_norm = normalized_auc(xs, ys, T2, ref=ref_ds, y0=0.0)    # optional normalized variant

            rows.append(dict(dataset=dataset, optimizer=mkey[0], zcp_method=mkey[1],
                             T1_min=T1, T2_min=T2, acc_T1=acc_T1, acc_T2=acc_T2,
                             auc_raw_T2=auc_raw, auc_norm_T2=auc_norm))
            auc_rows.append(dict(dataset=dataset, optimizer=mkey[0], zcp_method=mkey[1],
                                 T2_min=T2, auc_raw_T2=auc_raw, auc_norm_T2=auc_norm))

            # Legend text with raw AUC
            legend_texts[mkey] = f"{label} | AUC = {auc_raw:.3f}"

            # Plot lines/markers
            tgrid = np.linspace(0, xs[-1], 500)
            interp = np.interp(tgrid, xs, ys)
            ax.plot(tgrid, interp, color=color, linestyle=fmt, linewidth=2.5, label=label)
            ax.plot(xs, ys, linestyle="None", marker=marker, color=color, markersize=5, markeredgewidth=1)

        ax.set_xlabel("Wallclock Time (min) [Linear Scale]")
        ax.set_ylabel("Incumbent Search Validation Accuracy (%) [Linear Scale]")
        ax.set_title(f"HPO Incumbent Anytime Validation Performance | {dataset.upper()} | NAS-Bench-201")
        ax.set_xscale("linear"); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, which="both", ls="-", alpha=0.5)

        # legend in consistent order, with AUC numbers
        handles = [
            Line2D([0],[0],
                   color=method_to_color[m], linestyle=method_to_fmt[m], marker=method_to_marker[m],
                   linewidth=2.5, markersize=6,
                   label=legend_texts.get(m, format_method_label(m[0], m[1])))
            for m in method_list if (dataset, m) in per
        ]
        ax.legend(handles=handles, loc="center right")

        out_png = os.path.join(out_dir, f"hpo_anytime_{dataset}.png")
        plt.savefig(out_png, bbox_inches="tight"); plt.close()
        print(f"Saved {out_png}")

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "fixed_time_leaderboard.csv"), index=False)
        pd.DataFrame(auc_rows).to_csv(os.path.join(out_dir, "auc_summary.csv"), index=False)

def make_search_to_eval_transfer(metas, studies, out_dir, final_root_dir=None):
    # Build HPO best per dataset/method from Optuna (highest acc over all trials, all states)
    best_rows = []
    for meta, st in zip(metas, studies):
        vals = []
        for t in st.trials:
            v = trial_best_value(t)
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        best_rows.append(dict(dataset=meta["dataset"], optimizer=meta["optimizer"], zcp_method=meta["zcp_method"],
                              hpo_search_best=float(np.max(vals))))
    if not best_rows:
        print("No HPO best values.")
        return
    hpo_df = pd.DataFrame(best_rows).groupby(["dataset","optimizer","zcp_method"], dropna=False, as_index=False)["hpo_search_best"].max()

    # Load final runs from errors.json tree (no CSVs)
    final_df = load_final_runs(final_root_dir)
    if final_df is None:
        print("Skipping Search→Eval transfer (no final runs found).")
        return

    # Normalize None for merge and canonicalize optimizer names for join
    hpo_m = hpo_df.copy()
    fin_m = final_df.copy()
    SENT = "__NOZCP__"
    hpo_m["zcp_method"] = hpo_m["zcp_method"].fillna(SENT)
    fin_m["zcp_method"] = fin_m["zcp_method"].fillna(SENT)

    hpo_m["merge_opt"] = hpo_m["optimizer"].apply(canonical_optimizer_name)
    fin_m["merge_opt"] = fin_m["optimizer"].apply(canonical_optimizer_name)

    m = pd.merge(
        hpo_m, fin_m,
        left_on=["dataset","merge_opt","zcp_method"],
        right_on=["dataset","merge_opt","zcp_method"],
        how="inner"
    )
    if m.empty:
        print("No overlap between HPO and final runs (keys differ in dataset/optimizer/zcp_method).")
        return

    # restore None for reporting/labels
    m["zcp_method"] = m["zcp_method"].replace(SENT, np.nan)

    m["delta_final_minus_hpo"] = m["mean_final"] - m["hpo_search_best"]
    # Ranks per dataset
    m["rank_by_hpo"]   = m.groupby("dataset")["hpo_search_best"].rank(ascending=False, method="min")
    m["rank_by_final"] = m.groupby("dataset")["mean_final"].rank(ascending=False, method="min")
    m["rank_shift"]    = m["rank_by_final"] - m["rank_by_hpo"]

    m.sort_values(["dataset","mean_final"], ascending=[True, False]).to_csv(
        os.path.join(out_dir, "search_to_eval_transfer.csv"), index=False
    )

    # Use the SAME style mapping as the HPO plot
    method_list, method_to_color, _, method_to_marker = build_style_maps(metas)

    for ds, g in m.groupby("dataset"):
        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()

        for _, r in g.iterrows():
            zcp = None if (r["zcp_method"] is None or (isinstance(r["zcp_method"], float) and np.isnan(r["zcp_method"]))) else r["zcp_method"]
            mk = (r["optimizer_x"], zcp)
            color = method_to_color.get(mk, COLORS[0])
            marker= method_to_marker.get(mk, "o")
            ax.plot(r["hpo_search_best"], r["mean_final"], marker=marker, color=color,
                    linestyle="None", markersize=8, markeredgewidth=1,
                    label=format_method_label(mk[0], mk[1]))

        # y=x reference
        xmin = float(min(g["hpo_search_best"].min(), g["mean_final"].min()))
        xmax = float(max(g["hpo_search_best"].max(), g["mean_final"].max()))
        ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", color="gray", linewidth=1)

        # Legend with same method order
        present = {(r["optimizer_x"], None if (r["zcp_method"] is None or (isinstance(r["zcp_method"], float) and np.isnan(r["zcp_method"]))) else r["zcp_method"]) for _, r in g.iterrows()}
        handles = [
            Line2D([0],[0], marker=method_to_marker[m], color=method_to_color[m],
                   linestyle="None", markersize=8, label=format_method_label(m[0], m[1]))
            for m in method_list if m in present
        ]
        ax.legend(handles=handles, loc="lower right")  # align legend position

        # Axis style to mimic incumbent plots
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax.set_xlabel("HPO Search Validation Accuracy (%) [Linear Scale]")   # clearer x label
        ax.set_ylabel("Final Mean Validation Accuracy (%) [Linear Scale]")   # clearer y label
        ax.set_title(f"Search Phase -> Transfer -> Eval Phase  — {ds}")
        ax.grid(True, ls="-", alpha=0.5)
        out_png = os.path.join(out_dir, f"{ds}_search_to_eval_scatter.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db_dir", default="naslib/optimizers/oneshot/gsparsity/results_wide_hpo/WHPO_Databases", help="Folder with *.db from your HPO runs (WHPO_Databases)")
    ap.add_argument("--out_dir", default="naslib/optimizers/oneshot/gsparsity/plotting_scripts/plots/hpo")
    ap.add_argument("--search_space", default="nasbench201")
    ap.add_argument("--final_root_dir", default="naslib/optimizers/oneshot/gsparsity/result_final_hp", help="Root dir of final runs (errors.json tree) to build Search→Eval transfer")
    args = ap.parse_args()

    metas, studies = load_studies(args.db_dir, args.search_space)
    if not metas:
        print("No studies found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    make_fixed_time_reports(metas, studies, args.out_dir)
    make_search_to_eval_transfer(metas, studies, args.out_dir, final_root_dir=args.final_root_dir)

if __name__ == "__main__":
    main()