import os
import json
import time
import glob
import bisect
import argparse
from collections import defaultdict
import numpy as np
import csv

# Prefer NASLib helpers if available
try:
    from naslib.utils import get_dataset_api, get_project_root
except Exception:
    get_dataset_api = None
    get_project_root = None


def project_root():
    # Prefer NASLib helper but normalize to repo root (parent of the 'naslib' package)
    if get_project_root:
        try:
            p = os.path.abspath(str(get_project_root()))
            # If this points to the 'naslib' package dir, go one up to the repo root
            if os.path.basename(p) == "naslib":
                return os.path.dirname(p)
            return p
        except Exception:
            pass
    # Fallback: climb up from this file to the repo root (parent that contains 'naslib')
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(8):
        parent = os.path.dirname(cur)
        if os.path.isdir(os.path.join(parent, "naslib")):
            return parent
        cur = parent
    # Last resort: 4 levels up from this file
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))


def save_json(path, data, pretty=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if pretty:
            json.dump(data, f, indent=2, sort_keys=False)
        else:
            json.dump(data, f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _resolve_nb201_api(dataset_api_obj):
    # Unwrap potential wrappers
    api_like = dataset_api_obj
    for attr in ["nb201_api", "api", "nb201", "raw_api", "_api"]:
        if hasattr(dataset_api_obj, attr):
            api_like = getattr(dataset_api_obj, attr)
            break

    # Determine number of models
    n = None
    try:
        n = len(api_like)
    except Exception:
        pass
    if n is None:
        for cand in ["n_models", "n_archs", "num", "N", "n_architectures"]:
            if hasattr(api_like, cand):
                n = int(getattr(api_like, cand))
                break
    if n is None:
        n = 15625
    return api_like, n


def _query_meta_info(api_like, idx):
    # Try common NB201 methods
    for method in ["query_meta_info_by_index", "query_by_index", "arch", "get_arch"]:
        if hasattr(api_like, method):
            fn = getattr(api_like, method)
            try:
                return fn(idx, hp="200")
            except TypeError:
                return fn(idx)
    raise RuntimeError("Could not get NB201 meta_info by index.")


def _extract_acc(meta_info, dataset, split):
    """
    Extract accuracy for split in {'valid','train'} robustly.
    """
    modes_by_split = {
        "valid": ["x-valid", "valid", "val"],
        "train": ["train", "x-train"],
    }
    # Official NB201 pattern
    for mode in modes_by_split.get(split, []):
        try:
            metrics = meta_info.get_metrics(dataset, mode)
            if isinstance(metrics, dict):
                for key in ["accuracy", "acc", "top1", "best_acc", "best", "value"]:
                    if key in metrics:
                        return float(metrics[key])
                for subkey in ["eval", "metric", "stats"]:
                    if subkey in metrics and isinstance(metrics[subkey], dict):
                        for key in ["accuracy", "acc", "top1", "best_acc", "best"]:
                            if key in metrics[subkey]:
                                return float(metrics[subkey][key])
        except Exception:
            pass

    # Fallbacks
    for attr in ["metrics", "_metrics", "data", "results"]:
        if hasattr(meta_info, attr):
            try:
                obj = getattr(meta_info, attr)
                if isinstance(obj, dict) and dataset in obj:
                    ds_metrics = obj[dataset]
                    for key in [
                        f"{m}/accuracy" for m in modes_by_split.get(split, [])
                    ] + ["accuracy", "acc", "top1"]:
                        if key in ds_metrics:
                            return float(ds_metrics[key])
            except Exception:
                pass

    # Very last resort
    for cand in [
        f"{split}_acc",
        f"{split}acc",
        f"{'x_valid' if split == 'valid' else 'x_train'}_acc",
        "acc",
    ]:
        if hasattr(meta_info, cand):
            try:
                return float(getattr(meta_info, cand))
            except Exception:
                pass

    raise RuntimeError(f"Could not extract {split} accuracy for dataset={dataset}")


def build_nb201_distribution(dataset, out_dir, split):
    """
    split in {'valid','train'}
    """
    if get_dataset_api is None:
        raise RuntimeError(
            "NASLib get_dataset_api is not available in this environment."
        )

    api = get_dataset_api("nasbench201", dataset)

    # Fallback path: NASLib-style dict with 'nb201_data'
    if isinstance(api, dict) and "nb201_data" in api:
        ds_key = "cifar10-valid" if dataset in ["cifar10", "cifar10-valid"] else dataset
        rec_key = "eval_acc1es" if split == "valid" else "train_acc1es"

        accs, indices = [], []
        for idx, (_, arch_rec) in enumerate(api["nb201_data"].items()):
            try:
                series = arch_rec[ds_key][rec_key]
                # Use last epoch value
                if isinstance(series, (list, tuple)) and len(series) > 0:
                    acc = float(series[-1])
                else:
                    acc = float(series)
                indices.append(idx)
                accs.append(acc)
            except Exception:
                continue

        order = sorted(range(len(accs)), key=lambda i: accs[i])
        accs_sorted = [accs[i] for i in order]
        indices_sorted = [indices[i] for i in order]

        data = {
            "dataset": dataset,
            "split": split,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(accs_sorted),
            "indices_sorted": indices_sorted,
            "accs_sorted": accs_sorted,
            "accuracy_type": f"{split}_top1_percent",
        }

        save_path = os.path.join(out_dir, f"nb201_{split}_acc_rank_{dataset}.json")
        save_json(save_path, data, pretty=False)
        if data["total"] == 0:
            print(
                f"[WARN] Built empty NB201 distribution for {dataset}/{split} from nb201_data."
            )
        else:
            print(
                f"[INFO] Built NB201 distribution for {dataset}/{split}: {data['total']} entries."
            )
        return data, save_path

    # Original NATS-Bench API path
    api_like, n = _resolve_nb201_api(api)
    accs = []
    indices = []
    success, fail = 0, 0
    for idx in range(n):
        try:
            meta = _query_meta_info(api_like, idx)
            acc = _extract_acc(meta, dataset, split)
            indices.append(idx)
            accs.append(acc)
            success += 1
        except Exception:
            fail += 1
            continue

    order = sorted(range(len(accs)), key=lambda i: accs[i])
    accs_sorted = [accs[i] for i in order]
    indices_sorted = [indices[i] for i in order]

    data = {
        "dataset": dataset,
        "split": split,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(accs_sorted),
        "indices_sorted": indices_sorted,
        "accs_sorted": accs_sorted,
        "accuracy_type": f"{split}_top1_percent",
    }

    save_path = os.path.join(out_dir, f"nb201_{split}_acc_rank_{dataset}.json")
    save_json(save_path, data, pretty=False)
    print(
        f"[INFO] NB201 {dataset}/{split}: success={success}, fail={fail}, saved={len(accs_sorted)}"
    )
    if data["total"] == 0:
        print(
            f"[WARN] Built empty NB201 distribution for {dataset}/{split}. Check NB201 data availability."
        )
    return data, save_path


def load_or_build_distributions(datasets, out_dir, force=False):
    dist = {}
    for ds in datasets:
        dist[ds] = {}
        for split in ["valid", "train"]:
            path = os.path.join(out_dir, f"nb201_{split}_acc_rank_{ds}.json")
            need_rebuild = force
            if not need_rebuild and os.path.exists(path):
                try:
                    cached = load_json(path)
                    # Auto-rebuild if cached file is empty/invalid
                    if (
                        not isinstance(cached, dict)
                        or cached.get("total", 0) == 0
                        or not cached.get("accs_sorted")
                    ):
                        print(
                            f"[INFO] Rebuilding empty/invalid cached distribution: {path}"
                        )
                        need_rebuild = True
                    else:
                        dist[ds][split] = cached
                        continue
                except Exception:
                    need_rebuild = True
            if need_rebuild or not os.path.exists(path):
                dist[ds][split], _ = build_nb201_distribution(ds, out_dir, split)
    return dist


def percentile_from_distribution(value, accs_sorted):
    if not accs_sorted:
        return None
    pos = bisect.bisect_right(accs_sorted, value)
    return 100.0 * pos / float(len(accs_sorted))


def find_errors_json_files(base_dirs, dataset):
    """
    Accepts a list of base directories. Matches:
    <base>/*/nasbench201/<dataset>/<seed>/errors.json
    and <base>/*/*/nasbench201/<dataset>/<seed>/errors.json (keeps method_path segments).
    Returns list of dicts.
    """
    seen = set()
    results = []
    for base_dir in base_dirs:
        if not base_dir or not os.path.isdir(base_dir):
            continue
        pattern = os.path.join(
            base_dir, "**", "nasbench201", dataset, "*", "errors.json"
        )
        for fp in glob.glob(pattern, recursive=True):
            parts = fp.split(os.sep)
            try:
                idx_root = parts.index(os.path.basename(base_dir))
                idx_nb201 = parts.index("nasbench201")
            except ValueError:
                continue

            method_segments = parts[idx_root + 1 : idx_nb201]
            if not method_segments:
                continue
            method_path = "/".join(method_segments)
            ds = parts[idx_nb201 + 1]
            seed = parts[idx_nb201 + 2]
            if ds != dataset:
                continue

            key = (method_path, ds, seed, fp)
            if key in seen:
                continue
            seen.add(key)

            method = method_segments[0]
            zcp_name = method_segments[1] if len(method_segments) > 1 else None

            results.append(
                {
                    "method_path": method_path,
                    "method": method,
                    "zcp_method": zcp_name,
                    "dataset": ds,
                    "seed": seed,
                    "path": fp,
                }
            )
    return results


def _extract_list_of_floats(seq):
    vals = []
    if not isinstance(seq, list):
        return vals
    for item in seq:
        if isinstance(item, (int, float)):
            vals.append(float(item))
        elif isinstance(item, (list, tuple)) and len(item) > 0:
            # e.g., [x, y] -> take the last element as in your example
            try:
                vals.append(float(item[-1]))
            except Exception:
                continue
    return vals


def extract_queried_series(path):
    """
    Return both search-time and queried series.

    Keys in result:
      - 's_train', 's_val'           -> search-time train/val accuracies per epoch
      - 'q_train', 'q_val', 'q_test' -> queried evaluation accuracies per epoch
    """
    try:
        data = load_json(path)
    except Exception:
        return {}

    traj = (
        data
        if isinstance(data, dict)
        else (
            data[1]
            if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict)
            else None
        )
    )
    if not isinstance(traj, dict):
        return {}

    out = {}

    # Queried series (evaluation/benchmark)
    if "queried_train_acc" in traj:
        out["q_train"] = _extract_list_of_floats(traj["queried_train_acc"])
    if "queried_val_acc" in traj:
        out["q_val"] = _extract_list_of_floats(traj["queried_val_acc"])
    if "queried_test_acc" in traj:
        out["q_test"] = _extract_list_of_floats(traj["queried_test_acc"])

    # Search-time series (measured during search loop)
    if "train_acc" in traj:
        out["s_train"] = _extract_list_of_floats(traj["train_acc"])
    if "valid_acc" in traj:
        out["s_val"] = _extract_list_of_floats(traj["valid_acc"])
    elif "val_acc" in traj:  # some logs use val_acc
        out["s_val"] = _extract_list_of_floats(traj["val_acc"])

    return out


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return None
    r = np.corrcoef(x, y)[0, 1]
    return float(r)


def _rankdata(a):
    """
    Average ranks for ties, ranks start at 1.
    """
    a = np.asarray(a, dtype=float)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_vals = a[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    return pearson_corr(rx, ry)


def dump_csvs(per_seed, out_dir):
    # 1) Last-epoch summary per run
    per_run_csv = os.path.join(out_dir, "percentiles_per_run.csv")
    with open(per_run_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method_path",
                "method",
                "zcp_method",
                "dataset",
                "seed",
                "errors_json",
                # last queried values and percentiles
                "last_q_val_acc",
                "last_q_val_percentile",
                "last_q_train_acc",
                "last_q_train_percentile",
                # last search-time values
                "last_s_val_acc",
                "last_s_train_acc",
                # correlation (series)
                "pearson_s_val_vs_q_valpct",
                "spearman_s_val_vs_q_valpct",
                "pearson_s_trn_vs_q_trnpct",
                "spearman_s_trn_vs_q_trnpct",
                "pearson_s_val_vs_q_trnpct",
                "spearman_s_val_vs_q_trnpct",
                "pearson_s_trn_vs_q_valpct",
                "spearman_s_trn_vs_q_valpct",
            ]
        )
        for r in per_seed:
            last = r.get("last", {})
            corr = r.get("correlation", {})
            w.writerow(
                [
                    r.get("method_path"),
                    r.get("method"),
                    r.get("zcp_method"),
                    r.get("dataset"),
                    r.get("seed"),
                    r.get("errors_json"),
                    last.get("q_val_acc"),
                    last.get("q_val_percentile"),
                    last.get("q_train_acc"),
                    last.get("q_train_percentile"),
                    last.get("s_val_acc"),
                    last.get("s_train_acc"),
                    corr.get("pearson_s_val_vs_q_valpct"),
                    corr.get("spearman_s_val_vs_q_valpct"),
                    corr.get("pearson_s_trn_vs_q_trnpct"),
                    corr.get("spearman_s_trn_vs_q_trnpct"),
                    corr.get("pearson_s_val_vs_q_trnpct"),
                    corr.get("spearman_s_val_vs_q_trnpct"),
                    corr.get("pearson_s_trn_vs_q_valpct"),
                    corr.get("spearman_s_trn_vs_q_valpct"),
                ]
            )

    # 2) Per-epoch series per run
    per_epoch_csv = os.path.join(out_dir, "percentile_series_per_run.csv")
    with open(per_epoch_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method_path",
                "dataset",
                "seed",
                "epoch",
                # search-time raw series
                "s_train_value",
                "s_val_value",
                # queried raw + percentiles
                "q_train_value",
                "q_train_percentile",
                "q_val_value",
                "q_val_percentile",
            ]
        )
        for r in per_seed:
            ser = r.get("series", {})
            s_trn = ser.get("s_train_values", []) or []
            s_val = ser.get("s_val_values", []) or []
            q_trn = ser.get("q_train_values", []) or []
            q_trn_pct = ser.get("q_train_percentiles", []) or []
            q_val = ser.get("q_val_values", []) or []
            q_val_pct = ser.get("q_val_percentiles", []) or []
            max_len = max(len(s_trn), len(s_val), len(q_trn), len(q_val))
            for e in range(max_len):
                w.writerow(
                    [
                        r["method_path"],
                        r["dataset"],
                        r["seed"],
                        e,
                        s_trn[e] if e < len(s_trn) else None,
                        s_val[e] if e < len(s_val) else None,
                        q_trn[e] if e < len(q_trn) else None,
                        q_trn_pct[e] if e < len(q_trn_pct) else None,
                        q_val[e] if e < len(q_val) else None,
                        q_val_pct[e] if e < len(q_val_pct) else None,
                    ]
                )

    print(f"Wrote detailed CSVs:\n - {per_run_csv}\n - {per_epoch_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild NB201 distributions even if cache exists.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="",
        help="Override base directory that contains method/.../nasbench201/<dataset>/<seed>/errors.json",
    )
    parser.add_argument(
        "--print-details",
        action="store_true",
        help="Print per-run last-epoch percentiles to console.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Write CSVs with per-run and per-epoch individual values.",
    )
    args = parser.parse_args()

    root = project_root()
    # Datasets
    datasets = ["cifar100", "ImageNet16-120"]
    # Save distributions and reports near gsparsity code
    out_dir = os.path.join(
        root, "naslib", "optimizers", "oneshot", "gsparsity", "nb201_distributions"
    )

    # Candidate results roots (supports your configurator layouts)
    default_base = os.path.join(root, "naslib", "optimizers", "oneshot", "gsparsity")
    results_base_candidates = []
    if args.results_dir:
        # absolute or relative to repo root
        rb = (
            args.results_dir
            if os.path.isabs(args.results_dir)
            else os.path.join(root, args.results_dir)
        )
        results_base_candidates.append(os.path.abspath(rb))
    # Common subfolders you use
    for sub in ["result_final_hp", "test"]:
        results_base_candidates.append(os.path.join(default_base, sub))
    # Also allow scanning the whole gsparsity subtree as a fallback
    results_base_candidates.append(default_base)

    print(f"Project root: {root}")
    print(f"Saving NB201 distributions to: {out_dir}")
    print("Scanning results under:")
    for b in results_base_candidates:
        print(f" - {b}")

    # Step 1: Load or build distributions for valid and train
    distributions = load_or_build_distributions(datasets, out_dir, force=args.force)

    # Step 2: Scan runs and compute per-seed metrics
    per_seed = []
    for ds in datasets:
        files = find_errors_json_files(results_base_candidates, ds)
        if not files:
            print(f"No errors.json files found for dataset={ds}")
            continue

        accs_valid_sorted = distributions[ds]["valid"]["accs_sorted"]
        accs_train_sorted = distributions[ds]["train"]["accs_sorted"]

        print(f"Found {len(files)} errors.json files for dataset={ds}")

        for rec in files:
            series = extract_queried_series(rec["path"])
            if not series:
                continue

            # Separate search-time and queried series
            s_val = series.get("s_val", [])
            s_trn = series.get("s_train", [])
            q_val = series.get("q_val", [])
            q_trn = series.get("q_train", [])

            # Percentiles for queried series
            q_val_pct_series = [
                percentile_from_distribution(v, accs_valid_sorted) for v in q_val
            ]
            q_trn_pct_series = [
                percentile_from_distribution(v, accs_train_sorted) for v in q_trn
            ]

            # Last values
            last_q_val = q_val[-1] if q_val else None
            last_q_trn = q_trn[-1] if q_trn else None
            last_q_val_pct = (
                percentile_from_distribution(last_q_val, accs_valid_sorted)
                if last_q_val is not None
                else None
            )
            last_q_trn_pct = (
                percentile_from_distribution(last_q_trn, accs_train_sorted)
                if last_q_trn is not None
                else None
            )
            last_s_val = s_val[-1] if s_val else None
            last_s_trn = s_trn[-1] if s_trn else None

            # Per-epoch correlations between search-time raw values and queried percentiles
            corr = {}

            def _corr_xy(x, y, prefix):
                m = min(len(x), len(y))
                if m >= 2:
                    corr[f"pearson_{prefix}"] = pearson_corr(x[:m], y[:m])
                    corr[f"spearman_{prefix}"] = spearman_corr(x[:m], y[:m])
                else:
                    corr[f"pearson_{prefix}"] = None
                    corr[f"spearman_{prefix}"] = None

            _corr_xy(s_val, q_val_pct_series, "s_val_vs_q_valpct")
            _corr_xy(s_trn, q_trn_pct_series, "s_trn_vs_q_trnpct")
            _corr_xy(s_val, q_trn_pct_series, "s_val_vs_q_trnpct")  # cross
            _corr_xy(s_trn, q_val_pct_series, "s_trn_vs_q_valpct")  # cross

            per_seed.append(
                {
                    "method_path": rec["method_path"],
                    "method": rec["method"],
                    "zcp_method": rec["zcp_method"],
                    "dataset": rec["dataset"],
                    "seed": rec["seed"],
                    "errors_json": rec["path"],
                    "last": {
                        # queried last + percentile
                        "q_val_acc": last_q_val,
                        "q_val_percentile": last_q_val_pct,
                        "q_train_acc": last_q_trn,
                        "q_train_percentile": last_q_trn_pct,
                        # search-time last
                        "s_val_acc": last_s_val,
                        "s_train_acc": last_s_trn,
                    },
                    "series": {
                        # search
                        "s_val_values": s_val,
                        "s_train_values": s_trn,
                        # queried raw + percentiles
                        "q_val_values": q_val,
                        "q_val_percentiles": q_val_pct_series,
                        "q_train_values": q_trn,
                        "q_train_percentiles": q_trn_pct_series,
                    },
                    "correlation": corr,
                }
            )

    # Step 3: Aggregate by (method_path, dataset)
    grouped_val = defaultdict(list)
    grouped_train = defaultdict(list)
    grouped_runs = defaultdict(list)
    for r in per_seed:
        key = (r["method_path"], r["dataset"])
        grouped_runs[key].append(r)
        if r["last"]["q_val_percentile"] is not None:
            grouped_val[key].append(r["last"]["q_val_percentile"])
        if r["last"]["q_train_percentile"] is not None:
            grouped_train[key].append(r["last"]["q_train_percentile"])

    # Build summary with cross-seed correlations (last search vs last queried percentile)
    summary = []
    all_keys = (
        set(grouped_runs.keys()) | set(grouped_val.keys()) | set(grouped_train.keys())
    )
    for key in sorted(all_keys):
        method_path, dataset = key
        vals_v = grouped_val.get(key, [])
        vals_t = grouped_train.get(key, [])

        seeds_list = []
        s_val_last_list, q_valpct_last_list = [], []
        s_trn_last_list, q_trnpct_last_list = [], []

        for r in sorted(
            grouped_runs.get(key, []),
            key=lambda x: int(x["seed"]) if str(x["seed"]).isdigit() else x["seed"],
        ):
            last = r["last"]
            seeds_list.append(
                {
                    "seed": int(r["seed"]) if str(r["seed"]).isdigit() else r["seed"],
                    "q_val_percentile": last.get("q_val_percentile"),
                    "q_train_percentile": last.get("q_train_percentile"),
                    "q_val_acc": last.get("q_val_acc"),
                    "q_train_acc": last.get("q_train_acc"),
                    "s_val_acc": last.get("s_val_acc"),
                    "s_train_acc": last.get("s_train_acc"),
                }
            )
            if (
                last.get("s_val_acc") is not None
                and last.get("q_val_percentile") is not None
            ):
                s_val_last_list.append(last["s_val_acc"])
                q_valpct_last_list.append(last["q_val_percentile"])
            if (
                last.get("s_train_acc") is not None
                and last.get("q_train_percentile") is not None
            ):
                s_trn_last_list.append(last["s_train_acc"])
                q_trnpct_last_list.append(last["q_train_percentile"])

        # Cross-seed correlations (last epoch)
        corr_seed = {}
        if len(s_val_last_list) >= 2:
            corr_seed["pearson_last_s_val_vs_q_valpct"] = pearson_corr(
                s_val_last_list, q_valpct_last_list
            )
            corr_seed["spearman_last_s_val_vs_q_valpct"] = spearman_corr(
                s_val_last_list, q_valpct_last_list
            )
        else:
            corr_seed["pearson_last_s_val_vs_q_valpct"] = None
            corr_seed["spearman_last_s_val_vs_q_valpct"] = None
        if len(s_trn_last_list) >= 2:
            corr_seed["pearson_last_s_trn_vs_q_trnpct"] = pearson_corr(
                s_trn_last_list, q_trnpct_last_list
            )
            corr_seed["spearman_last_s_trn_vs_q_trnpct"] = spearman_corr(
                s_trn_last_list, q_trnpct_last_list
            )
        else:
            corr_seed["pearson_last_s_trn_vs_q_trnpct"] = None
            corr_seed["spearman_last_s_trn_vs_q_trnpct"] = None

        summary.append(
            {
                "method_path": method_path,
                "dataset": dataset,
                "num_seeds_val": len(vals_v),
                "mean_percentile_val": (sum(vals_v) / len(vals_v)) if vals_v else None,
                "num_seeds_train": len(vals_t),
                "mean_percentile_train": (sum(vals_t) / len(vals_t))
                if vals_t
                else None,
                "seeds": seeds_list,
                "cross_seed_correlation": corr_seed,
            }
        )

    # Save artifacts
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": datasets,
        "notes": [
            "Validation distributions are used to rank queried_val_acc (percentiles).",
            "Training distributions are used to rank queried_train_acc (percentiles).",
            "Per-run correlations relate search-time accuracies to queried percentiles.",
            "Cross-seed correlations use last-epoch search-time vs last-epoch queried percentiles.",
        ],
        "per_seed": per_seed,
        "summary_by_method_dataset": summary,
    }
    report_path = os.path.join(out_dir, "percentiles_and_correlation_summary.json")
    save_json(report_path, report, pretty=True)

    # Minimal console summary
    print("Mean percentile (last epoch) by method_path and dataset:")
    for item in sorted(summary, key=lambda x: (x["dataset"], x["method_path"])):
        mv = item["mean_percentile_val"]
        mt = item["mean_percentile_train"]
        print(
            f"- {item['dataset']:>14} | {item['method_path']:<40} | "
            f"seeds(val)={item['num_seeds_val']:>3} mean_val={mv:.2f}"
            if mv is not None
            else f"- {item['dataset']:>14} | {item['method_path']:<40} | seeds(val)={item['num_seeds_val']:>3} mean_val=None"
        )
        if mt is not None:
            print(
                f"  {'':>14} | {'':<40} | seeds(trn)={item['num_seeds_train']:>3} mean_trn={mt:.2f}"
            )
        else:
            print(
                f"  {'':>14} | {'':<40} | seeds(trn)={item['num_seeds_train']:>3} mean_trn=None"
            )
    print(f"\nSaved distributions and report to: {out_dir}")


if __name__ == "__main__":
    main()
