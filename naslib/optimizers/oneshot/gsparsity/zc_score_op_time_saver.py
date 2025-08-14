import argparse
import json
import os
import random
import subprocess
import sys
import time

import torch
import numpy as np
from fvcore.common.config import CfgNode

from naslib.utils import get_project_root, get_train_val_loaders
from naslib.search_spaces import NasBench201SearchSpace

# Optimizers
from naslib.optimizers.oneshot.gsparsity.zc_score_op_time_zcp_minmax_gsparse_optimizer import (
    ZCP_GSparseOptimizer as ZCPGSparse,
)
from naslib.optimizers.oneshot.gsparsity.zc_score_op_time_zc_pre_reducing_search_space_zcp_minmax_gsparse_optimizer import (
    ZCP_GSparseOptimizer as PreZCPZCPGSparse,
)


# Helpers (aligned to zc_score_op_time_saver.py)
def _collect_leaf_zcp(prim):
    """
    Collect leaf ZCPs exactly like loader/consumers expect:
      - prim.op[j] first (j-level)
      - then prim.op[j].op[k] (k-level) if present
      - fallback to prim itself if no subops
    """
    vals = []
    try:
        J = len(prim.op)
        # j-level
        for j in range(J):
            try:
                node = prim.op[j]
                if hasattr(node, "zero_cost_proxy"):
                    vals.append(float(node.zero_cost_proxy))
            except Exception:
                pass
        # k-level (after j-level, if present)
        for j in range(J):
            try:
                K = len(prim.op[j].op)
            except Exception:
                continue
            for k in range(K):
                node2 = prim.op[j].op[k]
                if hasattr(node2, "zero_cost_proxy"):
                    vals.append(float(node2.zero_cost_proxy))
    except AttributeError:
        # Simple primitive without children
        if hasattr(prim, "zero_cost_proxy"):
            vals.append(float(prim.zero_cost_proxy))
    return vals


def _collect_graph_scores(graph):
    scores = []
    ops = graph.get_all_edge_data(
        "op", scope=graph.OPTIMIZER_SCOPE, private_edge_data=True
    )
    for ei, mixed_op in enumerate(ops):
        prims = getattr(mixed_op, "primitives", [])
        for pj, prim in enumerate(prims):
            leaf_scores = _collect_leaf_zcp(prim)
            # Also capture primitive-level score if present (used after loading precomputed ZCPs)
            prim_score = getattr(prim, "zero_cost_proxy", None)
            prim_score = float(prim_score) if prim_score is not None else None
            scores.append(
                {
                    "edge_idx": ei,
                    "primitive_idx": pj,
                    "leaf_scores": leaf_scores,
                    "primitive_score": prim_score,
                }
            )
    return scores


def _agg_score_from_entry(entry):
    ls = entry.get("leaf_scores", []) or []
    if len(ls) > 0:
        return float(np.mean(ls))
    ps = entry.get("primitive_score", None)
    if ps is None:
        return None
    return float(ps)


def _verify_scores_agg_tol(
    saved_scores, currentScores, atol=1e-9, rtol=0.0, max_report=10
):
    mismatches = []
    if len(saved_scores) != len(currentScores):
        mismatches.append(
            f"Length mismatch: saved={len(saved_scores)} current={len(currentScores)}"
        )
        return mismatches
    for i, (s, c) in enumerate(zip(saved_scores, currentScores)):
        if s["edge_idx"] != c["edge_idx"] or s["primitive_idx"] != c["primitive_idx"]:
            mismatches.append(
                f"Index mismatch at idx {i}: saved(e{s['edge_idx']},p{s['primitive_idx']}) "
                f"!= current(e{c['edge_idx']},p{c['primitive_idx']})"
            )
            continue
        sa = _agg_score_from_entry(s)
        ca = _agg_score_from_entry(c)
        if sa is None and ca is None:
            continue
        if (sa is None) != (ca is None):
            mismatches.append(
                f"Aggregate presence mismatch at edge {s['edge_idx']} prim {s['primitive_idx']}: "
                f"saved={sa} current={ca}"
            )
            continue
        if not (abs(sa - ca) <= atol + rtol * abs(ca)):
            mismatches.append(
                f"Aggregate mismatch at edge {s['edge_idx']} prim {s['primitive_idx']}: "
                f"saved={sa} current={ca}"
            )
        if len(mismatches) >= max_report:
            break
    return mismatches


def _verify_scores(saved_scores, currentScores, max_report=10):
    mismatches = []
    if len(saved_scores) != len(currentScores):
        mismatches.append(
            f"Length mismatch: saved={len(saved_scores)} current={len(currentScores)}"
        )
        return mismatches

    for i, (s, c) in enumerate(zip(saved_scores, currentScores)):
        if s["edge_idx"] != c["edge_idx"] or s["primitive_idx"] != c["primitive_idx"]:
            mismatches.append(
                f"Index mismatch at idx {i}: saved(e{s['edge_idx']},p{s['primitive_idx']}) "
                f"!= current(e{c['edge_idx']},p{c['primitive_idx']})"
            )
            continue

        # Strict leaf-level comparison (every sub-operation)
        ls, lc = s["leaf_scores"], c["leaf_scores"]
        if len(ls) != len(lc):
            mismatches.append(
                f"Leaf length mismatch at edge {s['edge_idx']} prim {s['primitive_idx']}: "
                f"saved={len(ls)} current={len(lc)}"
            )
            if len(mismatches) >= max_report:
                break
            continue
        for k, (a, b) in enumerate(zip(ls, lc)):
            if a != b:
                mismatches.append(
                    f"Leaf value mismatch at edge {s['edge_idx']} prim {s['primitive_idx']} leaf {k}: "
                    f"saved={a} current={b}"
                )
                break

        if len(mismatches) >= max_report:
            break

        # Strict primitive-level comparison as well (must match exactly including None)
        if s.get("primitive_score", None) != c.get("primitive_score", None):
            mismatches.append(
                f"Primitive score mismatch at edge {s['edge_idx']} prim {s['primitive_idx']}: "
                f"saved={s.get('primitive_score')} current={c.get('primitive_score')}"
            )
            if len(mismatches) >= max_report:
                break

    return mismatches


def _verify_scores_tol(saved_scores, currentScores, atol=0, rtol=0, max_report=10):
    mismatches = []
    if len(saved_scores) != len(currentScores):
        mismatches.append(
            f"Length mismatch: saved={len(saved_scores)} current={len(currentScores)}"
        )
        return mismatches
    for i, (s, c) in enumerate(zip(saved_scores, currentScores)):
        if s["edge_idx"] != c["edge_idx"] or s["primitive_idx"] != c["primitive_idx"]:
            mismatches.append(
                f"Index mismatch at idx {i}: saved(e{s['edge_idx']},p{s['primitive_idx']}) "
                f"!= current(e{c['edge_idx']},p{c['primitive_idx']})"
            )
            continue
        ls, lc = s["leaf_scores"], c["leaf_scores"]
        if len(ls) != len(lc):
            mismatches.append(
                f"Leaf length mismatch at edge {s['edge_idx']} prim {s['primitive_idx']}: "
                f"saved={len(ls)} current={len(lc)}"
            )
            continue
        for k, (a, b) in enumerate(zip(ls, lc)):
            if not (abs(a - b) <= atol + rtol * abs(b)):
                mismatches.append(
                    f"Value mismatch at edge {s['edge_idx']} prim {s['primitive_idx']} leaf {k}: "
                    f"saved={a} current={b}"
                )
                break
        if len(mismatches) >= max_report:
            break
    return mismatches


def _nb201_classes(dataset):
    return {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}[dataset]


def _build_min_config(dataset, seed, zcp_method, pre_op_dir=None, pre_arch_base=None):
    """
    pre_op_dir: directory containing file 'op_scores_{dataset}_{zcp_method}_seed{seed}.json'
    pre_arch_base: base path without suffix (e.g., .../arch_scores), loader appends _{dataset}.json
    """
    cfg = {
        "data": str(get_project_root()) + "/data",
        "dataset": dataset,
        "dataset_subset": 1.0,
        "search": {
            "seed": seed,
            "batch_size": 64,
            "train_portion": 0.95,
            "cutout": False,
            "grad_clip": None,
            "weight_decay": 60.0,
            "threshold": 1e-6,
            "normalization": "div",
            "normalization_exponent": 0.5,
            "learning_rate": 1e-3,
            "momentum": 0.8,
            "learning_rate_min": 1e-4,
            "zcp_method": zcp_method,
            "train_workers": 0,
            "val_workers": 0,
        },
    }
    if pre_op_dir:
        cfg["search"]["pre_computed_op_zc_scores_dir"] = pre_op_dir
    if pre_arch_base:
        cfg["search"]["pre_computed_zc_scores"] = pre_arch_base
    return CfgNode.load_cfg(json.dumps(cfg))


def _run_precompute_once(
    dataset, zcp_method, seed, out_json, disable_prune, arch_scores_dir, prune_topk
):
    """
    Compute per-op ZCPs and write:
      - op_scores_{dataset}_{zcp_method}_seed{seed}.json
      - op_scores_duration_{dataset}_{zcp_method}_seed{seed}.json
    We time only the scoring + normalization (recorded by the optimizer).
    """
    use_pruned = not disable_prune

    # Build config: no precomputed op scores (we want to compute); use arch scores for pruning if requested
    pre_arch_base = os.path.join(arch_scores_dir, "arch_scores") if use_pruned else None
    config = _build_min_config(
        dataset, seed, zcp_method, pre_op_dir=None, pre_arch_base=pre_arch_base
    )

    # Data + search space
    train_loader, _, _, _, _ = get_train_val_loaders(
        config, train_workers=0, val_workers=0
    )
    n_cls = _nb201_classes(dataset)
    search_space = NasBench201SearchSpace(n_classes=n_cls)

    # Instantiate optimizer
    if use_pruned:
        opt = PreZCPZCPGSparse(config)
        opt.adapt_search_space(
            search_space=search_space, train_loader=train_loader, resume_from_path=None
        )
    else:
        opt = ZCPGSparse(config)
        opt.adapt_search_space(search_space=search_space, train_loader=train_loader)

    # Collect scores from graph (already normalized)
    graph = opt.graph
    saved_scores = _collect_graph_scores(graph)

    # Write scores JSON
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "meta": {
            "dataset": dataset,
            "zcp_method": zcp_method,
            "seed": seed,
            "normalized": True,
            "pruned": use_pruned,
        },
        "scores": saved_scores,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f)

    # Write duration JSON using optimizer-recorded timing (scoring + normalization only)
    duration = float(getattr(opt, "_zcp_compute_time", 0.0))
    dur_path = out_json.replace("op_scores_", "op_scores_duration_")
    with open(dur_path, "w") as f:
        json.dump({"duration": duration}, f)

    print(f"[SAVE] Scores -> {out_json}")
    print(f"[TIME] Saved op precompute duration {duration:.3f}s -> {dur_path}")


def _verify_loaded_matches_saved(
    saved_json,
    dataset,
    zcp_method,
    seed,
    use_pruned,
    pre_dir,
    arch_scores_dir,
    report_json,
):
    with open(saved_json, "r") as f:
        payload = json.load(f)
    saved_scores = payload["scores"]

    # Choose optimizer and config
    if use_pruned:
        # PreZCPZCP: loads arch-level pruning and applies precomputed op scores
        pre_arch_base = os.path.join(arch_scores_dir, "arch_scores")
        config = _build_min_config(
            dataset, seed, zcp_method, pre_op_dir=pre_dir, pre_arch_base=pre_arch_base
        )
        opt = PreZCPZCPGSparse(config)
    else:
        # ZCPGSparse: unpruned, but loads precomputed op scores
        config = _build_min_config(
            dataset, seed, zcp_method, pre_op_dir=pre_dir, pre_arch_base=None
        )
        opt = ZCPGSparse(config)

    # Data + search space
    train_loader, _, _, _, _ = get_train_val_loaders(
        config, train_workers=0, val_workers=0
    )
    n_cls = _nb201_classes(dataset)
    search_space = NasBench201SearchSpace(n_classes=n_cls)

    # Adapt search space (this will load and apply precomputed op scores)
    if use_pruned:
        opt.adapt_search_space(
            search_space=search_space, train_loader=train_loader, resume_from_path=None
        )
    else:
        opt.adapt_search_space(search_space=search_space, train_loader=train_loader)

    graph = opt.graph
    currentScores = _collect_graph_scores(graph)
    # Strict, exact comparison: every primitive and each sub-op must match, no tolerance
    mismatches = _verify_scores(saved_scores, currentScores)

    os.makedirs(os.path.dirname(report_json), exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(
            {
                "ok": len(mismatches) == 0,
                "mismatches_count": len(mismatches),
                "first_mismatches": mismatches[:10],
                "saved_meta": payload.get("meta", {}),
            },
            f,
            indent=2,
        )

    status = "OK" if len(mismatches) == 0 else f"{len(mismatches)} mismatch(es)"
    print(
        f"[VERIFY] {('PRUNED' if use_pruned else 'UNPRUNED')} -> {status}. Report: {report_json}"
    )


def _verify_recompute_matches_saved(
    saved_json, dataset, zcp_method, seed, use_pruned, arch_scores_dir, report_json
):
    with open(saved_json, "r") as f:
        payload = json.load(f)
    saved_scores = payload["scores"]

    # Build config WITHOUT precomputed op scores to force fresh computation
    if use_pruned:
        pre_arch_base = os.path.join(arch_scores_dir, "arch_scores")
        config = _build_min_config(
            dataset, seed, zcp_method, pre_op_dir=None, pre_arch_base=pre_arch_base
        )
        opt = PreZCPZCPGSparse(config)
    else:
        config = _build_min_config(
            dataset, seed, zcp_method, pre_op_dir=None, pre_arch_base=None
        )
        opt = ZCPGSparse(config)

    train_loader, _, _, _, _ = get_train_val_loaders(
        config, train_workers=0, val_workers=0
    )
    n_cls = _nb201_classes(dataset)
    search_space = NasBench201SearchSpace(n_classes=n_cls)

    if use_pruned:
        opt.adapt_search_space(
            search_space=search_space, train_loader=train_loader, resume_from_path=None
        )
    else:
        opt.adapt_search_space(search_space=search_space, train_loader=train_loader)

    currentScores = _collect_graph_scores(opt.graph)
    # Strict, exact comparison here as well
    mismatches = _verify_scores(saved_scores, currentScores)

    os.makedirs(os.path.dirname(report_json), exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(
            {
                "ok": len(mismatches) == 0,
                "mismatches_count": len(mismatches),
                "first_mismatches": mismatches[:10],
                "saved_meta": payload.get("meta", {}),
                "check": "recompute_vs_saved",
            },
            f,
            indent=2,
        )
    print(
        f"[VERIFY-RECOMPUTE] {('PRUNED' if use_pruned else 'UNPRUNED')} -> "
        f"{'OK' if len(mismatches) == 0 else f'{len(mismatches)} mismatch(es)'}; Report: {report_json}"
    )


def _verify_graphs_match(graph_a, graph_b, max_report=10):
    """
    Strict, exact equality by collecting scores from both graphs via the same i->j->k traversal.
    """
    a = _collect_graph_scores(graph_a)
    b = _collect_graph_scores(graph_b)
    return _verify_scores(a, b, max_report=max_report)


def _verify_loaded_vs_recomputed(
    dataset, zcp_method, seed, use_pruned, pre_dir, arch_scores_dir, report_json
):
    """
    Build two graphs:
      - one that LOADS precomputed op ZCPs
      - one that RECOMPUTES op ZCPs
    Then compare their leaf and primitive-level scores exactly.
    """
    # Configs
    if use_pruned:
        pre_arch_base = os.path.join(arch_scores_dir, "arch_scores")
        cfg_loaded = _build_min_config(
            dataset, seed, zcp_method, pre_op_dir=pre_dir, pre_arch_base=pre_arch_base
        )
        cfg_recomp = _build_min_config(
            dataset, seed, zcp_method, pre_op_dir=None, pre_arch_base=pre_arch_base
        )
        OptLoaded = PreZCPZCPGSparse
        OptRecomp = PreZCPZCPGSparse
    else:
        cfg_loaded = _build_min_config(
            dataset, seed, zcp_method, pre_op_dir=pre_dir, pre_arch_base=None
        )
        cfg_recomp = _build_min_config(
            dataset, seed, zcp_method, pre_op_dir=None, pre_arch_base=None
        )
        OptLoaded = ZCPGSparse
        OptRecomp = ZCPGSparse

    # Data + search space
    train_loader, _, _, _, _ = get_train_val_loaders(
        cfg_loaded, train_workers=0, val_workers=0
    )
    n_cls = _nb201_classes(dataset)
    ss_loaded = NasBench201SearchSpace(n_classes=n_cls)
    ss_recomp = NasBench201SearchSpace(n_classes=n_cls)

    # Adapt search spaces
    opt_loaded = OptLoaded(cfg_loaded)
    opt_recomp = OptRecomp(cfg_recomp)
    if use_pruned:
        opt_loaded.adapt_search_space(
            search_space=ss_loaded, train_loader=train_loader, resume_from_path=None
        )
        opt_recomp.adapt_search_space(
            search_space=ss_recomp, train_loader=train_loader, resume_from_path=None
        )
    else:
        opt_loaded.adapt_search_space(search_space=ss_loaded, train_loader=train_loader)
        opt_recomp.adapt_search_space(search_space=ss_recomp, train_loader=train_loader)

    # Strict equality comparison of both graphs
    mismatches = _verify_graphs_match(opt_loaded.graph, opt_recomp.graph)

    os.makedirs(os.path.dirname(report_json), exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(
            {
                "ok": len(mismatches) == 0,
                "mismatches_count": len(mismatches),
                "first_mismatches": mismatches[:10],
                "meta": {
                    "dataset": dataset,
                    "zcp_method": zcp_method,
                    "seed": seed,
                    "pruned": use_pruned,
                    "check": "loaded_vs_recomputed",
                },
            },
            f,
            indent=2,
        )
    print(
        f"[VERIFY L-vs-R] {('PRUNED' if use_pruned else 'UNPRUNED')} -> "
        f"{'OK' if len(mismatches) == 0 else f'{len(mismatches)} mismatch(es)'}; Report: {report_json}"
    )


def main():
    ap = argparse.ArgumentParser(
        "Precompute per-op ZCPs (unpruned/pruned) and verify loading."
    )
    ap.add_argument(
        "--dataset", required=True, choices=["cifar10", "cifar100", "ImageNet16-120"]
    )
    ap.add_argument(
        "--zcp_method",
        required=True,
        choices=[
            "params",
            "jacov",
            "synflow",
            "grad_norm",
            "grasp",
            "nwot",
            "epe_nas",
            "zen",
            "flops",
        ],
    )
    ap.add_argument("--seed", type=int, default=2152435495)
    ap.add_argument(
        "--arch_scores_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor",
        help="Directory containing arch_scores_{dataset}.json",
    )
    ap.add_argument(
        "--out_base_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_op_scoring_timefactor",
        help="Base directory; will write into {base}/unpruned and {base}/pruned",
    )
    ap.add_argument("--prune_topk", type=int, default=750)
    ap.add_argument("--skip_unpruned", action="store_true")
    ap.add_argument("--skip_pruned", action="store_true")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    # NEW: enforce deterministic CUDA math
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    # Prepare paths
    base = args.out_base_dir
    unpruned_dir = os.path.join(base, "unpruned")
    pruned_dir = os.path.join(base, "pruned")
    os.makedirs(unpruned_dir, exist_ok=True)
    os.makedirs(pruned_dir, exist_ok=True)

    base_name = f"op_scores_{args.dataset}_{args.zcp_method}_seed{args.seed}.json"
    unpruned_json = os.path.join(unpruned_dir, base_name)
    pruned_json = os.path.join(pruned_dir, base_name)

    # 1) Precompute unpruned (no pruning)
    if not args.skip_unpruned:
        _run_precompute_once(
            dataset=args.dataset,
            zcp_method=args.zcp_method,
            seed=args.seed,
            out_json=unpruned_json,
            disable_prune=True,
            arch_scores_dir=args.arch_scores_dir,
            prune_topk=args.prune_topk,
        )

    # 2) Precompute pruned (using arch_scores to prune before scoring)
    if not args.skip_pruned:
        _run_precompute_once(
            dataset=args.dataset,
            zcp_method=args.zcp_method,
            seed=args.seed,
            out_json=pruned_json,
            disable_prune=False,
            arch_scores_dir=args.arch_scores_dir,
            prune_topk=args.prune_topk,
        )

    # 3) Verify loading for UNPRUNED
    if (not args.skip_unpruned) and os.path.exists(unpruned_json):
        unpruned_report = os.path.join(
            unpruned_dir,
            f"verify_loaded_vs_saved_{args.dataset}_{args.zcp_method}_seed{args.seed}.json",
        )
        _verify_loaded_matches_saved(
            saved_json=unpruned_json,
            dataset=args.dataset,
            zcp_method=args.zcp_method,
            seed=args.seed,
            use_pruned=False,
            pre_dir=unpruned_dir,
            arch_scores_dir=args.arch_scores_dir,
            report_json=unpruned_report,
        )
    else:
        print(
            f"[SKIP] Unpruned verification skipped (skip flag or missing file: {unpruned_json})"
        )

    # 4) Verify loading for PRUNED
    if (not args.skip_pruned) and os.path.exists(pruned_json):
        pruned_report = os.path.join(
            pruned_dir,
            f"verify_loaded_vs_saved_{args.dataset}_{args.zcp_method}_seed{args.seed}.json",
        )
        _verify_loaded_matches_saved(
            saved_json=pruned_json,
            dataset=args.dataset,
            zcp_method=args.zcp_method,
            seed=args.seed,
            use_pruned=True,
            pre_dir=pruned_dir,
            arch_scores_dir=args.arch_scores_dir,
            report_json=pruned_report,
        )
    else:
        print(
            f"[SKIP] Pruned verification skipped (skip flag or missing file: {pruned_json})"
        )

    # Recompute checks
    if (not args.skip_unpruned) and os.path.exists(unpruned_json):
        unpruned_recomp = os.path.join(
            unpruned_dir,
            f"verify_recompute_vs_saved_{args.dataset}_{args.zcp_method}_seed{args.seed}.json",
        )
        _verify_recompute_matches_saved(
            saved_json=unpruned_json,
            dataset=args.dataset,
            zcp_method=args.zcp_method,
            seed=args.seed,
            use_pruned=False,
            arch_scores_dir=args.arch_scores_dir,
            report_json=unpruned_recomp,
        )
    else:
        print(
            f"[SKIP] Unpruned recompute check skipped (skip flag or missing file: {unpruned_json})"
        )

    if (not args.skip_pruned) and os.path.exists(pruned_json):
        pruned_recomp = os.path.join(
            pruned_dir,
            f"verify_recompute_vs_saved_{args.dataset}_{args.zcp_method}_seed{args.seed}.json",
        )
        _verify_recompute_matches_saved(
            saved_json=pruned_json,
            dataset=args.dataset,
            zcp_method=args.zcp_method,
            seed=args.seed,
            use_pruned=True,
            arch_scores_dir=args.arch_scores_dir,
            report_json=pruned_recomp,
        )
    else:
        print(
            f"[SKIP] Pruned recompute check skipped (skip flag or missing file: {pruned_json})"
        )

    # Loaded vs Recomputed cross-check (graph-to-graph)
    if (not args.skip_unpruned) and os.path.exists(unpruned_json):
        unpruned_lr = os.path.join(
            unpruned_dir,
            f"verify_loaded_vs_recomputed_{args.dataset}_{args.zcp_method}_seed{args.seed}.json",
        )
        _verify_loaded_vs_recomputed(
            dataset=args.dataset,
            zcp_method=args.zcp_method,
            seed=args.seed,
            use_pruned=False,
            pre_dir=unpruned_dir,
            arch_scores_dir=args.arch_scores_dir,
            report_json=unpruned_lr,
        )
    if (not args.skip_pruned) and os.path.exists(pruned_json):
        pruned_lr = os.path.join(
            pruned_dir,
            f"verify_loaded_vs_recomputed_{args.dataset}_{args.zcp_method}_seed{args.seed}.json",
        )
        _verify_loaded_vs_recomputed(
            dataset=args.dataset,
            zcp_method=args.zcp_method,
            seed=args.seed,
            use_pruned=True,
            pre_dir=pruned_dir,
            arch_scores_dir=args.arch_scores_dir,
            report_json=pruned_lr,
        )


if __name__ == "__main__":
    main()
