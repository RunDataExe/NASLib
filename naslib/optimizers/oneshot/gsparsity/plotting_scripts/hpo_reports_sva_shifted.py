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

DEFAULTS = [
    (0.9411764705882353, 0.27450980392156865, 0.7490196078431373),  # [240,70,191]
    (0.3843137254901961, 0.7686274509803922, 0.20784313725490197),  # [98,196,53]
    (0.35294117647058826, 0.25098039215686274, 0.6549019607843137),  # [90,64,167]
    (0.43529411764705883, 0.8627450980392157, 0.5607843137254902),  # [111,220,143]
    (0.6313725490196078, 0.0784313725490196, 0.27450980392156865),  # [161,20,70]
    (0.00784313725490196, 0.807843137254902, 0.9803921568627451),  # [2,206,250]
    (0.9529411764705882, 0.3607843137254902, 0.1568627450980392),  # [243,92,40]
    (0.2196078431372549, 0.3607843137254902, 0.12941176470588237),  # [56,92,33]
    (1.0, 0.592156862745098, 0.4196078431372549),  # [255,151,107]
    (0.7019607843137254, 0.7803921568627451, 0.5647058823529412),  # [179,199,144]
    (0.7098039215686275, 0.5529411764705883, 0.0),  # [181,141,0]
]
C_MAX = len(DEFAULTS)
COLORS = [*DEFAULTS[:C_MAX]] * 3
FMTS = [*["-"] * C_MAX, *["--"] * C_MAX, *[":"] * C_MAX]
MARKERS = ["o", "s", "+", "D", "x", "^", "*", "v", "<", ">", "p", "h", "H", "P"]

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"

# ---- ZCP pre methods and durations loader (align with check_time_shifted_hp_choice.py) ----
ZCP_PRE_METHODS = {
    "zcp-pre_gsparsity",
    "zcp-pre_zcp_gsparsity",
}


def load_dataset_durations(durations_dir: str) -> dict:
    """
    Loads durations from arch_scores_duration_*.json files into a mapping:
    {dataset_name: duration_seconds}
    """
    if not durations_dir or not os.path.isdir(durations_dir):
        return {}
    mapping = {}
    pattern = os.path.join(durations_dir, "arch_scores_duration_*.json")
    for path in glob.glob(pattern):
        base = os.path.basename(path)
        dataset = base.replace("arch_scores_duration_", "").replace(".json", "")
        try:
            with open(path, "r") as f:
                d = json.load(f)
            dur = float(d.get("duration", 0.0))
            mapping[dataset] = dur
        except Exception:
            continue
    return mapping


def _sort_key_method(m):
    # m is (optimizer, zcp_method); map None/NaN to '' for stable sorting
    return (
        m[0] or "",
        "" if (m[1] is None or (isinstance(m[1], float) and np.isnan(m[1]))) else m[1],
    )


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
    o = str(opt)
    # Specific → general to avoid collapsing pre variants
    if "zcp-pre_zcp_gsparsity" in o:
        return "zcp-pre_zcp_gsparsity"
    if "zcp-pre_gsparsity" in o:
        return "zcp-pre_gsparsity"
    if "zcp_gsparsity" in o:
        return "zcp_gsparsity"
    if "gsparsity" in o and "zcp" not in o:
        return "gsparsity"
    if "darts" in o:
        return "darts"
    return o


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
    method_list = sorted(
        {(m["optimizer"], m["zcp_method"]) for m in metas}, key=_sort_key_method
    )
    method_to_color = {m: COLORS[i % len(COLORS)] for i, m in enumerate(method_list)}
    method_to_fmt = {m: FMTS[i % len(FMTS)] for i, m in enumerate(method_list)}
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


# ---- helpers for time spans and group reference horizons (T_ref) ----
def _study_timespan(trials):
    starts = [t.datetime_start for t in trials if t.datetime_start is not None]
    ends = [t.datetime_complete for t in trials if t.datetime_complete is not None]
    if not starts or not ends:
        return None, None, None
    first_start = min(starts)
    last_end = max(ends)
    return first_start, last_end, (last_end - first_start).total_seconds()


def _compute_group_ref_horizons(metas, studies):
    """
    Build T_ref per (dataset, seed) group:
    - T_ref = max span among non-zcp-pre methods in that group.
    - If none, T_ref = max span among all methods in that group.
    """
    # collect spans
    group_spans_all = defaultdict(list)
    group_spans_nonpre = defaultdict(list)
    for meta, st in zip(metas, studies):
        fs, le, span = _study_timespan(st.trials)
        if span is None:
            continue
        key = (meta["dataset"], meta["seed"])
        group_spans_all[key].append(span)
        if meta["optimizer"] not in ZCP_PRE_METHODS:
            group_spans_nonpre[key].append(span)

    tref = {}
    for key in set(group_spans_all.keys()):
        if group_spans_nonpre.get(key):
            tref[key] = float(max(group_spans_nonpre[key]))
        else:
            tref[key] = float(max(group_spans_all[key]))
    return tref


def best_over_time(trials):
    # Staircase incumbent vs wall-clock seconds (trial completion)
    starts = [t.datetime_start for t in trials if t.datetime_start]
    if not starts:
        return np.array([]), np.array([])
    t0 = min(starts)
    events = []
    for t in trials:
        v = trial_best_value(t)
        if v is None or t.datetime_complete is None:
            continue
        # use seconds (no division by 60)
        x = (t.datetime_complete - t0).total_seconds()
        events.append((x, v))
    if not events:
        return np.array([]), np.array([])

    events.sort()
    xs, ys = [0.0], [0.0]  # ensure all curves start at 0 and ramp to first datapoint
    best = -np.inf
    for x, v in events:
        best = max(best, v)
        xs.append(float(x))
        ys.append(float(best))
    return np.asarray(xs), np.asarray(ys)


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
            st = optuna.load_study(
                storage=f"sqlite:///{db}", study_name=meta["study_name"]
            )
        except Exception as e:
            print(f"Skip {db}: {e}")
            continue
        metas.append(meta)
        studies.append(st)
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
                files.append(
                    dict(
                        path=p,
                        optimizer=optimizer,
                        zcp_method=zcp_method,
                        dataset=dataset,
                        search_space=search_space,
                        seed=seed,
                    )
                )
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
            rows.append(
                {
                    "dataset": f["dataset"],
                    "optimizer": f["optimizer"],
                    "zcp_method": f["zcp_method"],  # may be None
                    "seed": f["seed"],
                    "final_acc": float(qva[-1]),
                }
            )
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # keep None groups (dropna=False)
    agg = df.groupby(
        ["dataset", "optimizer", "zcp_method"], dropna=False, as_index=False
    )["final_acc"].agg(mean_final="mean", std_final="std", seeds="count")
    return agg


# ---- Reports ----
def make_fixed_time_reports(metas, studies, out_dir, durations_map=None):
    os.makedirs(out_dir, exist_ok=True)
    durations_map = durations_map or {}

    # Build method style mapping (consistent order across plots)
    method_list, method_to_color, method_to_fmt, method_to_marker = build_style_maps(
        metas
    )

    # Compute group reference horizons T_ref per (dataset, seed)
    group_tref = _compute_group_ref_horizons(metas, studies)

    # Build curves grouped by dataset/method, apply shift+clip for zcp-pre only
    per = defaultdict(list)
    for meta, st in zip(metas, studies):
        xs, ys = best_over_time(st.trials)
        if len(xs) == 0:
            continue

        if meta["optimizer"] in ZCP_PRE_METHODS:
            # per-dataset offset (if missing, treat as 0)
            offset = float(durations_map.get(meta["dataset"], 0.0))

            # Apply offset only to points after time 0, keep the start at 0
            if offset > 0.0 and len(xs) >= 2:
                xs = xs.copy()
                xs[1:] = xs[1:] + offset

            # clip to group T_ref so zcp-pre "search" + offset doesn't exceed others' horizon
            tref = group_tref.get((meta["dataset"], meta["seed"]))
            if tref is not None:
                mask = xs <= tref
                # Require at least one datapoint beyond t=0 to keep the run
                if np.sum(mask) <= 1:
                    continue
                xs = xs[mask]
                ys = ys[mask]

        per[(meta["dataset"], method_key(meta))].append(dict(xs=xs, ys=ys, meta=meta))

    # Decide common horizons per dataset (unchanged; used for metrics tables only)
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
            fmt = method_to_fmt[mkey]
            marker = method_to_marker[mkey]
            label = format_method_label(mkey[0], mkey[1])

            # Use the first run (as in original script)
            xs = runs[0]["xs"]
            ys = runs[0]["ys"]

            # Metrics within dataset
            acc_T1 = incumbent_at(xs, ys, T1)
            acc_T2 = incumbent_at(xs, ys, T2)
            auc_raw = normalized_auc(xs, ys, T2, ref=None, y0=0.0)
            auc_norm = normalized_auc(xs, ys, T2, ref=ref_ds, y0=0.0)

            rows.append(
                dict(
                    dataset=dataset,
                    optimizer=mkey[0],
                    zcp_method=mkey[1],
                    T1_sec=T1,
                    T2_sec=T2,
                    acc_T1=acc_T1,
                    acc_T2=acc_T2,
                    auc_raw_T2=auc_raw,
                    auc_norm_T2=auc_norm,
                )
            )
            auc_rows.append(
                dict(
                    dataset=dataset,
                    optimizer=mkey[0],
                    zcp_method=mkey[1],
                    T2_sec=T2,
                    auc_raw_T2=auc_raw,
                    auc_norm_T2=auc_norm,
                )
            )

            # Legend text with raw AUC
            legend_texts[mkey] = f"{label} | AUC = {auc_raw:.2f}"

            # Plot lines/markers
            tgrid = np.linspace(0, xs[-1], 500)
            interp = np.interp(tgrid, xs, ys)
            ax.plot(
                tgrid, interp, color=color, linestyle=fmt, linewidth=2.5, label=label
            )
            ax.plot(
                xs,
                ys,
                linestyle="None",
                marker=marker,
                color=color,
                markersize=5,
                markeredgewidth=1,
            )

        ax.set_xlabel("Wallclock Time (s) [Linear Scale]")
        ax.set_ylabel("Incumbent Search Validation Accuracy (%) [Linear Scale]")
        ax.set_title(
            f"HPO Incumbent Anytime Search Validation Performance | {dataset.upper()} | NAS-Bench-201"
        )
        ax.set_xscale("linear")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(True, which="both", ls="-", alpha=0.5)

        # legend in consistent order, with AUC numbers
        handles = [
            Line2D(
                [0],
                [0],
                color=method_to_color[m],
                linestyle=method_to_fmt[m],
                marker=method_to_marker[m],
                linewidth=2.5,
                markersize=6,
                label=legend_texts.get(m, format_method_label(m[0], m[1])),
            )
            for m in method_list
            if (dataset, m) in per
        ]
        ax.legend(handles=handles, loc="center right")

        out_png = os.path.join(out_dir, f"hpo_anytime_{dataset}.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_png}")

    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "fixed_time_leaderboard.csv"), index=False
        )
        pd.DataFrame(auc_rows).to_csv(
            os.path.join(out_dir, "auc_summary.csv"), index=False
        )


def make_search_to_eval_transfer(
    metas,
    studies,
    out_dir,
    final_root_dir=None,
    annot_pos="above_legend",
    durations_map=None,
):
    durations_map = durations_map or {}

    # Compute group reference horizons T_ref per (dataset, seed)
    group_tref = _compute_group_ref_horizons(metas, studies)

    # Build HPO best per dataset/method from Optuna
    best_rows = []
    for meta, st in zip(metas, studies):
        # For non zcp-pre: best over all trials (as before)
        if meta["optimizer"] not in ZCP_PRE_METHODS:
            vals = []
            for t in st.trials:
                v = trial_best_value(t)
                if v is not None:
                    vals.append(v)
            if not vals:
                continue
            best_rows.append(
                dict(
                    dataset=meta["dataset"],
                    optimizer=meta["optimizer"],
                    zcp_method=meta["zcp_method"],
                    hpo_search_best=float(np.max(vals)),
                )
            )
            continue

        # For zcp-pre: apply cutoff using offset and T_ref (same logic as the checker)
        # Determine first_start
        starts = [t.datetime_start for t in st.trials if t.datetime_start is not None]
        first_start = min(starts) if starts else None
        tref = group_tref.get((meta["dataset"], meta["seed"]))
        offset_sec = float(durations_map.get(meta["dataset"], 0.0))

        if first_start is None or tref is None:
            # fallback: no timing => skip or take overall best
            vals = []
            for t in st.trials:
                v = trial_best_value(t)
                if v is not None:
                    vals.append(v)
            if not vals:
                continue
            best_rows.append(
                dict(
                    dataset=meta["dataset"],
                    optimizer=meta["optimizer"],
                    zcp_method=meta["zcp_method"],
                    hpo_search_best=float(np.max(vals)),
                )
            )
        else:
            if offset_sec >= tref:
                cutoff = first_start  # no eligible trial completes <= cutoff later than start
                # pick nothing -> skip if no trial eligible
            else:
                cutoff = first_start + pd.to_timedelta(tref - offset_sec, unit="s")

            # select best among eligible
            eligible_vals = []
            for t in st.trials:
                if t.datetime_complete is None:
                    continue
                if t.datetime_complete <= cutoff:
                    v = trial_best_value(t)
                    if v is not None:
                        eligible_vals.append(v)
            if not eligible_vals:
                # if nothing eligible, skip this study in best_rows
                continue
            best_rows.append(
                dict(
                    dataset=meta["dataset"],
                    optimizer=meta["optimizer"],
                    zcp_method=meta["zcp_method"],
                    hpo_search_best=float(np.max(eligible_vals)),
                )
            )

    if not best_rows:
        print("No HPO best values.")
        return
    hpo_df = (
        pd.DataFrame(best_rows)
        .groupby(["dataset", "optimizer", "zcp_method"], dropna=False, as_index=False)[
            "hpo_search_best"
        ]
        .max()
    )

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

    # Canonical method key
    hpo_m["merge_opt"] = hpo_m["optimizer"].apply(canonical_optimizer_name)
    fin_m["merge_opt"] = fin_m["optimizer"].apply(canonical_optimizer_name)

    # COLLAPSE to a single row per (dataset, canonical-method, zcp) BEFORE merge
    hpo_m = hpo_m.groupby(
        ["dataset", "merge_opt", "zcp_method"], dropna=False, as_index=False
    ).agg(hpo_search_best=("hpo_search_best", "max"))
    fin_m = fin_m.groupby(
        ["dataset", "merge_opt", "zcp_method"], dropna=False, as_index=False
    ).agg(
        mean_final=("mean_final", "mean"),
        std_final=("std_final", "mean"),
        seeds=("seeds", "sum"),
    )

    # Merge on canonical method key
    m = pd.merge(
        hpo_m,
        fin_m,
        on=["dataset", "merge_opt", "zcp_method"],
        how="inner",
    )
    if m.empty:
        print(
            "No overlap between HPO and final runs (keys differ in dataset/optimizer/zcp_method)."
        )
        return

    # restore None for reporting/labels and set a stable optimizer column for plotting
    m["zcp_method"] = m["zcp_method"].replace(SENT, np.nan)
    m["optimizer_x"] = m["merge_opt"]

    # FINAL DEDUP: keep a single point per (dataset, canonical-method, zcp)
    # rule: prefer higher HPO-best (falls back to higher final mean if desired)
    m = m.sort_values(
        ["dataset", "merge_opt", "zcp_method", "hpo_search_best", "mean_final"],
        ascending=[True, True, True, False, False],
    ).drop_duplicates(subset=["dataset", "merge_opt", "zcp_method"], keep="first")

    m["delta_final_minus_hpo"] = m["mean_final"] - m["hpo_search_best"]
    # Ranks per dataset
    m["rank_by_hpo"] = m.groupby("dataset")["hpo_search_best"].rank(
        ascending=False, method="min"
    )
    m["rank_by_final"] = m.groupby("dataset")["mean_final"].rank(
        ascending=False, method="min"
    )
    m["rank_shift"] = m["rank_by_final"] - m["rank_by_hpo"]

    m.sort_values(["dataset", "mean_final"], ascending=[True, False]).to_csv(
        os.path.join(out_dir, "search_to_eval_transfer.csv"), index=False
    )

    # --- Print per-dataset ranks and proxy-fit metrics (HPO vs Final) ---
    # For each dataset show optimizer, zcp, HPO best, Final mean, delta and ranks.
    rank_rows = []
    for ds, gds in m.groupby("dataset"):
        print(f"\nDataset: {ds}")
        print(
            "Optimizer\tZCP\tHPO_best\tFinal_mean\tΔ(Final−HPO)\tRank_HPO\tRank_Final\tRank_shift"
        )
        for _, r in gds.sort_values("rank_by_final", ascending=True).iterrows():
            z = (
                r["zcp_method"]
                if not (
                    isinstance(r["zcp_method"], float) and np.isnan(r["zcp_method"])
                )
                else ""
            )
            print(
                f"{r['optimizer_x']}\t{z}\t{r['hpo_search_best']:.4f}\t{r['mean_final']:.4f}\t{r['delta_final_minus_hpo']:.4f}\t"
                f"{int(r['rank_by_hpo'])}\t{int(r['rank_by_final'])}\t{float(r['rank_shift']):.2f}"
            )
            rank_rows.append(
                {
                    "dataset": ds,
                    "optimizer": r["optimizer_x"],
                    "zcp_method": r["zcp_method"],
                    "hpo_search_best": float(r["hpo_search_best"]),
                    "mean_final": float(r["mean_final"]),
                    "delta_final_minus_hpo": float(r["delta_final_minus_hpo"]),
                    "rank_by_hpo": int(r["rank_by_hpo"]),
                    "rank_by_final": int(r["rank_by_final"]),
                    "rank_shift": float(r["rank_shift"]),
                }
            )

        # Proxy-fit metrics
        x = gds["hpo_search_best"].astype(float).to_numpy()
        y = gds["mean_final"].astype(float).to_numpy()
        mae = float(np.nanmean(np.abs(y - x))) if x.size > 0 else np.nan
        rmse = float(np.sqrt(np.nanmean((y - x) ** 2))) if x.size > 0 else np.nan
        pearson = np.nan
        if x.size >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            pearson = float(np.corrcoef(x, y)[0, 1])
        print(
            f"Proxy-fit: Pearson r = {pearson if np.isfinite(pearson) else 'n/a'}, MAE = {mae:.4f}, RMSE = {rmse:.4f}"
        )

    # --- Cross-dataset print: CIFAR100 (HPO) -> CIFAR10 (Final) ---
    # Uses precomputed hpo_m/fin_m with canonical optimizer keys; print only (no plot/result changes).
    try:
        hpo_m_lc = hpo_m.copy()
        fin_m_lc = fin_m.copy()
        hpo_m_lc["dataset_lc"] = hpo_m_lc["dataset"].astype(str).str.lower()
        fin_m_lc["dataset_lc"] = fin_m_lc["dataset"].astype(str).str.lower()

        hm = hpo_m_lc[hpo_m_lc["dataset_lc"] == "cifar100"].copy()
        fm = fin_m_lc[fin_m_lc["dataset_lc"] == "cifar10"].copy()

        if not hm.empty and not fm.empty:
            mc = pd.merge(
                hm,
                fm,
                on=["merge_opt", "zcp_method"],
                how="inner",
                suffixes=("_hpo", "_eval"),
            )
            if not mc.empty:
                # Restore None for zcp display
                mc["zcp_method"] = mc["zcp_method"].replace(SENT, np.nan)
                mc["optimizer_x"] = mc["merge_opt"]
                mc["delta_final_minus_hpo"] = mc["mean_final"] - mc["hpo_search_best"]

                # Ranks within eval dataset (single eval dataset: CIFAR10)
                mc["rank_by_hpo"] = mc["hpo_search_best"].rank(
                    ascending=False, method="min"
                )
                mc["rank_by_final"] = mc["mean_final"].rank(
                    ascending=False, method="min"
                )
                mc["rank_shift"] = mc["rank_by_final"] - mc["rank_by_hpo"]

                print("\nCross-dataset: CIFAR100 (HPO) -> CIFAR10 (Final)")
                print(
                    "Optimizer\tZCP\tHPO_best[CIFAR100]\tFinal_mean[CIFAR10]\tΔ(Final−HPO)\tRank_HPO\tRank_Final\tRank_shift"
                )

                cross_rows = []
                for _, r in mc.sort_values("rank_by_final", ascending=True).iterrows():
                    z = (
                        r["zcp_method"]
                        if not (
                            isinstance(r["zcp_method"], float)
                            and np.isnan(r["zcp_method"])
                        )
                        else ""
                    )
                    print(
                        f"{r['optimizer_x']}\t{z}\t{r['hpo_search_best']:.4f}\t{r['mean_final']:.4f}\t{r['delta_final_minus_hpo']:.4f}\t"
                        f"{int(r['rank_by_hpo'])}\t{int(r['rank_by_final'])}\t{float(r['rank_shift']):.2f}"
                    )
                    cross_rows.append(
                        {
                            "optimizer": r["optimizer_x"],
                            "zcp_method": r["zcp_method"],
                            "hpo_best_cifar100": float(r["hpo_search_best"]),
                            "final_mean_cifar10": float(r["mean_final"]),
                            "delta_final_minus_hpo": float(r["delta_final_minus_hpo"]),
                            "rank_by_hpo": int(r["rank_by_hpo"]),
                            "rank_by_final": int(r["rank_by_final"]),
                            "rank_shift": float(r["rank_shift"]),
                        }
                    )

                # Save cross-dataset CSV
                if cross_rows:
                    pd.DataFrame(cross_rows).to_csv(
                        os.path.join(out_dir, "sva_ranks_cifar100_to_cifar10.csv"),
                        index=False,
                    )

                # Proxy-fit for cross transfer
                x = mc["hpo_search_best"].astype(float).to_numpy()
                y = mc["mean_final"].astype(float).to_numpy()
                mae = float(np.nanmean(np.abs(y - x))) if x.size > 0 else np.nan
                rmse = (
                    float(np.sqrt(np.nanmean((y - x) ** 2))) if x.size > 0 else np.nan
                )
                pearson = np.nan
                if x.size >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                    pearson = float(np.corrcoef(x, y)[0, 1])
                print(
                    f"Proxy-fit (cross): Pearson r = {pearson if np.isfinite(pearson) else 'n/a'}, MAE = {mae:.4f}, RMSE = {rmse:.4f}"
                )
            else:
                print("\nCross-dataset: CIFAR100->CIFAR10 has no overlapping methods.")
        else:
            print(
                "\nCross-dataset: Missing data for CIFAR100 (HPO) or CIFAR10 (Final)."
            )
    except Exception as e:
        print(f"\nCross-dataset print failed: {e}")

    # Save a CSV with the rank rows and proxy-fit aggregated per dataset
    if rank_rows:
        pd.DataFrame(rank_rows).to_csv(
            os.path.join(out_dir, "sva_ranks_per_dataset.csv"), index=False
        )

    # Use the SAME style mapping as the HPO plot
    method_list, method_to_color, _, method_to_marker = build_style_maps(metas)

    for ds, g in m.groupby("dataset"):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        for _, r in g.iterrows():
            zcp = (
                None
                if (
                    r["zcp_method"] is None
                    or (
                        isinstance(r["zcp_method"], float) and np.isnan(r["zcp_method"])
                    )
                )
                else r["zcp_method"]
            )
            mk = (r["optimizer_x"], zcp)
            color = method_to_color.get(mk, COLORS[0])
            marker = method_to_marker.get(mk, "o")
            ax.plot(
                r["hpo_search_best"],
                r["mean_final"],
                marker=marker,
                color=color,
                linestyle="None",
                markersize=8,
                markeredgewidth=1,
                label=format_method_label(mk[0], mk[1]),
            )

        # y=x reference
        xmin = float(min(g["hpo_search_best"].min(), g["mean_final"].min()))
        xmax = float(max(g["hpo_search_best"].max(), g["mean_final"].max()))
        ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", color="gray", linewidth=1)

        # ---- quantitative annotations (no slope line) ----
        x = g["hpo_search_best"].astype(float).to_numpy()
        y = g["mean_final"].astype(float).to_numpy()
        pearson = np.nan
        if x.size >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            pearson = float(np.corrcoef(x, y)[0, 1])
        rx = pd.Series(x).rank(method="average").to_numpy()
        ry = pd.Series(y).rank(method="average").to_numpy()
        spearman = np.nan
        if rx.size >= 2 and np.nanstd(rx) > 0 and np.nanstd(ry) > 0:
            spearman = float(np.corrcoef(rx, ry)[0, 1])

        delta_mean = float(g["delta_final_minus_hpo"].mean())
        delta_std = float(g["delta_final_minus_hpo"].std(ddof=0))
        avg_rank_shift = float(g["rank_shift"].mean())
        rank_shift_std = float(g["rank_shift"].std(ddof=0))
        lines = [
            f"Pearson r = {pearson:.2f}" if np.isfinite(pearson) else "Pearson r = n/a",
            f"Spearman ρ = {spearman:.2f}"
            if np.isfinite(spearman)
            else "Spearman ρ = n/a",
            f"Mean ± SD of ΔAcc (Final_mean − HPO_best): {delta_mean:.2f} ± {delta_std:.2f}",
            f"Mean ± SD of Rank Shift: {avg_rank_shift:.2f} ± {rank_shift_std:.2f}",
        ]
        txt = "\n".join(lines)

        # Legend with same method order
        present = {
            (
                r["optimizer_x"],
                None
                if (
                    r["zcp_method"] is None
                    or (
                        isinstance(r["zcp_method"], float) and np.isnan(r["zcp_method"])
                    )
                )
                else r["zcp_method"],
            )
            for _, r in g.iterrows()
        }
        handles = [
            Line2D(
                [0],
                [0],
                marker=method_to_marker[m],
                color=method_to_color[m],
                linestyle="None",
                markersize=8,
                label=format_method_label(m[0], m[1]),
            )
            for m in method_list
            if m in present
        ]
        leg = ax.legend(handles=handles, loc="lower right")

        # ---- place annotation either above legend or at bottom-left ----
        if annot_pos == "bottom_left":
            ax.text(
                0.02,
                0.02,
                txt,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="gray",
                    linewidth=0.5,
                ),
            )
        else:
            # above_legend
            fig.canvas.draw()  # ensure we have a renderer
            bbox_disp = leg.get_window_extent(renderer=fig.canvas.get_renderer())
            bbox_ax = bbox_disp.transformed(ax.transAxes.inverted())
            # place just above the legend, right-aligned to its right edge
            x = min(bbox_ax.x1, 0.98)
            y = min(bbox_ax.y1 + 0.02, 0.98)
            ax.text(
                x,
                y,
                txt,
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="gray",
                    linewidth=0.5,
                ),
            )

        # Axis style
        ax.set_xscale("linear")
        ax.set_yscale("linear")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        ax.set_xlabel("HPO Search Validation Accuracy (%) [Linear Scale]")
        ax.set_ylabel("Final Mean Validation Accuracy (%) [Linear Scale]")
        ax.set_title(f"HPO Phase -> Transfer -> Eval Phase | {ds}")
        ax.grid(True, ls="-", alpha=0.5)
        out_png = os.path.join(out_dir, f"{ds}_search_to_eval_scatter.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_png}")

    # ---- Additional cross-dataset transfer: CIFAR100 (HPO) -> CIFAR10 (Eval) ----
    # Prepare case-insensitive dataset keys for joining while preserving originals
    hpo_m_lc = hpo_m.copy()
    fin_m_lc = fin_m.copy()
    if "dataset" in hpo_m_lc.columns:
        hpo_m_lc["dataset_lc"] = hpo_m_lc["dataset"].astype(str).str.lower()
    if "dataset" in fin_m_lc.columns:
        fin_m_lc["dataset_lc"] = fin_m_lc["dataset"].astype(str).str.lower()

    cross_pairs = [("cifar100", "cifar10")]
    for src, dst in cross_pairs:
        hm = hpo_m_lc[hpo_m_lc.get("dataset_lc", "") == src].copy()
        fm = fin_m_lc[fin_m_lc.get("dataset_lc", "") == dst].copy()
        if hm.empty or fm.empty:
            print(f"No data for cross transfer {src} -> {dst}.")
            continue

        mc = pd.merge(
            hm,
            fm,
            left_on=["merge_opt", "zcp_method"],
            right_on=["merge_opt", "zcp_method"],
            how="inner",
            suffixes=("_hpo", "_eval"),
        )
        if mc.empty:
            print(f"No overlap between HPO({src}) and Eval({dst}) runs after merge.")
            continue

        # Restore None for zcp_method for reporting/labels
        mc["zcp_method"] = mc["zcp_method"].replace(SENT, np.nan)

        # Harmonize plotting columns
        mc["optimizer_x"] = mc["merge_opt"]  # canonical optimizer key for plotting
        mc["hpo_dataset"] = mc.get("dataset_hpo", src)
        mc["eval_dataset"] = mc.get("dataset_eval", dst)

        # FINAL DEDUP: keep a single point per (eval_dataset, canonical-method, zcp)
        mc = mc.sort_values(
            [
                "eval_dataset",
                "merge_opt",
                "zcp_method",
                "hpo_search_best",
                "mean_final",
            ],
            ascending=[True, True, True, False, False],
        ).drop_duplicates(
            subset=["eval_dataset", "merge_opt", "zcp_method"], keep="first"
        )

        # Metrics and ranks (per eval dataset)
        mc["delta_final_minus_hpo"] = mc["mean_final"] - mc["hpo_search_best"]
        mc["rank_by_hpo"] = mc.groupby("eval_dataset")["hpo_search_best"].rank(
            ascending=False, method="min"
        )
        mc["rank_by_final"] = mc.groupby("eval_dataset")["mean_final"].rank(
            ascending=False, method="min"
        )
        mc["rank_shift"] = mc["rank_by_final"] - mc["rank_by_hpo"]

        # Save CSV
        csv_path = os.path.join(out_dir, f"search_to_eval_transfer_{src}_to_{dst}.csv")
        mc.sort_values(["eval_dataset", "mean_final"], ascending=[True, False]).to_csv(
            csv_path, index=False
        )
        print(f"Saved {csv_path}")

        # Plot per eval dataset
        method_list, method_to_color, _, method_to_marker = build_style_maps(metas)
        for ds, g in mc.groupby("eval_dataset"):
            fig = plt.figure(figsize=(8, 6))
            ax = plt.gca()

            for _, r in g.iterrows():
                zcp = (
                    None
                    if (
                        r["zcp_method"] is None
                        or (
                            isinstance(r["zcp_method"], float)
                            and np.isnan(r["zcp_method"])
                        )
                    )
                    else r["zcp_method"]
                )
                mk = (r["optimizer_x"], zcp)
                color = method_to_color.get(mk, COLORS[0])
                marker = method_to_marker.get(mk, "o")
                ax.plot(
                    r["hpo_search_best"],
                    r["mean_final"],
                    marker=marker,
                    color=color,
                    linestyle="None",
                    markersize=8,
                    markeredgewidth=1,
                    label=format_method_label(mk[0], mk[1]),
                )

            xmin = float(min(g["hpo_search_best"].min(), g["mean_final"].min()))
            xmax = float(max(g["hpo_search_best"].max(), g["mean_final"].max()))
            ax.plot(
                [xmin, xmax], [xmin, xmax], linestyle="--", color="gray", linewidth=1
            )

            x = g["hpo_search_best"].astype(float).to_numpy()
            y = g["mean_final"].astype(float).to_numpy()
            pearson = np.nan
            if x.size >= 2 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
                pearson = float(np.corrcoef(x, y)[0, 1])
            rx = pd.Series(x).rank(method="average").to_numpy()
            ry = pd.Series(y).rank(method="average").to_numpy()
            spearman = np.nan
            if rx.size >= 2 and np.nanstd(rx) > 0 and np.nanstd(ry) > 0:
                spearman = float(np.corrcoef(rx, ry)[0, 1])

            delta_mean = float(g["delta_final_minus_hpo"].mean())
            delta_std = float(g["delta_final_minus_hpo"].std(ddof=0))
            avg_rank_shift = float(g["rank_shift"].mean())
            rank_shift_std = float(g["rank_shift"].std(ddof=0))

            lines = [
                f"Pearson r = {pearson:.2f}"
                if np.isfinite(pearson)
                else "Pearson r = n/a",
                f"Spearman ρ = {spearman:.2f}"
                if np.isfinite(spearman)
                else "Spearman ρ = n/a",
                f"Mean ± SD of ΔAcc (Final_mean − HPO_best): {delta_mean:.2f} ± {delta_std:.2f}",
                f"Mean ± SD of Rank Shift: {avg_rank_shift:.2f} ± {rank_shift_std:.2f}",
            ]
            txt = "\n".join(lines)

            present = {
                (
                    r["optimizer_x"],
                    None
                    if (
                        r["zcp_method"] is None
                        or (
                            isinstance(r["zcp_method"], float)
                            and np.isnan(r["zcp_method"])
                        )
                    )
                    else r["zcp_method"],
                )
                for _, r in g.iterrows()
            }
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker=method_to_marker[m],
                    color=method_to_color[m],
                    linestyle="None",
                    markersize=8,
                    label=format_method_label(m[0], m[1]),
                )
                for m in method_list
                if m in present
            ]
            leg = ax.legend(handles=handles, loc="lower right")

            if annot_pos == "bottom_left":
                ax.text(
                    0.02,
                    0.02,
                    txt,
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.85,
                        edgecolor="gray",
                        linewidth=0.5,
                    ),
                )
            else:
                fig.canvas.draw()
                bbox_disp = leg.get_window_extent(renderer=fig.canvas.get_renderer())
                bbox_ax = bbox_disp.transformed(ax.transAxes.inverted())
                x_ = min(bbox_ax.x1, 0.98)
                y_ = min(bbox_ax.y1 + 0.02, 0.98)
                ax.text(
                    x_,
                    y_,
                    txt,
                    transform=ax.transAxes,
                    va="bottom",
                    ha="right",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.85,
                        edgecolor="gray",
                        linewidth=0.5,
                    ),
                )

            ax.set_xscale("linear")
            ax.set_yscale("linear")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            ax.set_xlabel("HPO Search Validation Accuracy (%) [Linear Scale]")
            ax.set_ylabel("Final Mean Validation Accuracy (%) [Linear Scale]")
            src_lbl = str(g["hpo_dataset"].iloc[0]).upper()
            dst_lbl = str(ds).upper()
            ax.set_title(
                f"HPO Phase -> Transfer -> Eval Phase | HPO: {src_lbl} -> Eval: {dst_lbl}"
            )
            ax.grid(True, ls="-", alpha=0.5)

            out_png = os.path.join(
                out_dir, f"{src}_to_{dst}_search_to_eval_scatter.png"
            )
            plt.savefig(out_png, bbox_inches="tight")
            plt.close()
            print(f"Saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db_dir",
        default="naslib/optimizers/oneshot/gsparsity/results_wide_hpo/WHPO_Databases",
        help="Folder with *.db from your HPO runs (WHPO_Databases)",
    )
    ap.add_argument(
        "--out_dir",
        default="naslib/optimizers/oneshot/gsparsity/plotting_scripts/plots/hpo",
    )
    ap.add_argument("--search_space", default="nasbench201")
    ap.add_argument(
        "--final_root_dir",
        default="naslib/optimizers/oneshot/gsparsity/result_final_hp",
        help="Root dir of final runs (errors.json tree) to build Search→Eval transfer",
    )
    # NEW: where to place the annotation
    ap.add_argument(
        "--annot_pos", choices=["above_legend", "bottom_left"], default="above_legend"
    )
    # NEW: durations dir to load arch scoring offsets (same as check_time_shifted_hp_choice.py)
    ap.add_argument(
        "--durations_dir",
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor",
        help="Directory containing arch_scores_duration_*.json files for per-dataset ZCP pre offsets.",
    )
    args = ap.parse_args()

    metas, studies = load_studies(args.db_dir, args.search_space)
    if not metas:
        print("No studies found.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    durations_map = load_dataset_durations(args.durations_dir)

    make_fixed_time_reports(metas, studies, args.out_dir, durations_map=durations_map)
    make_search_to_eval_transfer(
        metas,
        studies,
        args.out_dir,
        final_root_dir=args.final_root_dir,
        annot_pos=args.annot_pos,
        durations_map=durations_map,
    )


if __name__ == "__main__":
    main()
