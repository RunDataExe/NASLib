import argparse
import json
import math
import os
import random
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

# --- HELPER AND VERIFICATION FUNCTIONS (MOVED FROM SAVER) ---


def _collect_graph_scores(graph):
    name_to_mod = dict(graph.named_modules())
    mod_to_name = {id(m): n for n, m in name_to_mod.items()}

    def mod_name(m):
        return mod_to_name.get(id(m), None)

    scores = []
    ops = graph.get_all_edge_data(
        "op", scope=graph.OPTIMIZER_SCOPE, private_edge_data=True
    )
    for ei, mixed_op in enumerate(ops):
        prims = getattr(mixed_op, "primitives", [])
        for pj, prim in enumerate(prims):
            leaf_scores, leaf_modules = [], []
            try:
                # Case for nested Sequential ops
                for node in prim.op:
                    if hasattr(node, "zero_cost_proxy"):
                        leaf_scores.append(float(node.zero_cost_proxy))
                    leaf_modules.append(mod_name(node))
            except AttributeError:
                # Case for simple ops
                if hasattr(prim, "zero_cost_proxy"):
                    leaf_scores.append(float(prim.zero_cost_proxy))
                leaf_modules.append(mod_name(prim))

            prim_score = getattr(prim, "zero_cost_proxy", None)
            scores.append(
                {
                    "edge_idx": ei,
                    "primitive_idx": pj,
                    "primitive_module": mod_name(prim),
                    "leaf_scores": leaf_scores,
                    "leaf_modules": leaf_modules,
                    "primitive_score": float(prim_score)
                    if prim_score is not None
                    else None,
                }
            )
    return scores


def _verify_scores(saved_scores, currentScores, max_report=10, atol=1e-9, rtol=1e-7):
    cur_by_name = {c.get("primitive_module"): c for c in currentScores}
    mismatches = []
    for s in saved_scores:
        key = s.get("primitive_module")
        c = cur_by_name.get(key)
        if c is None:
            mismatches.append(f"Missing primitive_module in current: {key}")
            if len(mismatches) >= max_report:
                break
            continue

        ls, lc = s.get("leaf_scores", []), c.get("leaf_scores", [])
        lsn, lcn = s.get("leaf_modules", []), c.get("leaf_modules", [])
        if lsn != lcn:
            mismatches.append(
                f"Leaf module list mismatch for primitive {key}: saved={lsn} current={lcn}"
            )
            if len(mismatches) >= max_report:
                break
            continue

        for i, (a, b) in enumerate(zip(ls, lc)):
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                mismatches.append(
                    f"Leaf value mismatch for primitive {key} leaf {i}: saved={a} current={b}"
                )
                break
        if len(mismatches) >= max_report:
            break
    return mismatches


def _verify_graphs_match(graph_a, graph_b, max_report=10):
    a = _collect_graph_scores(graph_a)
    b = _collect_graph_scores(graph_b)
    return _verify_scores(a, b, max_report=max_report)


def _nb201_classes(dataset):
    return {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}[dataset]


def _build_min_config(dataset, seed, zcp_method, pre_op_dir=None, pre_arch_base=None):
    cfg = {
        "data": str(get_project_root()) + "/data",
        "dataset": dataset,
        "search": {
            "seed": seed,
            "grad_clip": 0,
            "weight_decay": 60,
            "threshold": 0.000001,
            "normalization": "div",
            "normalization_exponent": 0.5,
            "learning_rate": 0.0001,
            "momentum": 0.8,
            "learning_rate_min": 0.0001,
            "batch_size": 64,
            "train_portion": 0.95,
            "cutout": False,
            "cutout_length": 16,
            "zcp_method": zcp_method,
        },
    }
    if pre_op_dir:
        cfg["search"]["pre_computed_op_zc_scores_dir"] = pre_op_dir
    if pre_arch_base:
        cfg["search"]["pre_computed_zc_scores"] = pre_arch_base
    return CfgNode.load_cfg(json.dumps(cfg))


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

    pre_arch_base = os.path.join(arch_scores_dir, "arch_scores") if use_pruned else None
    config = _build_min_config(
        dataset, seed, zcp_method, pre_op_dir=pre_dir, pre_arch_base=pre_arch_base
    )
    opt = PreZCPZCPGSparse(config) if use_pruned else ZCPGSparse(config)

    train_loader, _, _, _, _ = get_train_val_loaders(
        config, train_workers=0, val_workers=0
    )
    search_space = NasBench201SearchSpace(n_classes=_nb201_classes(dataset))

    opt.adapt_search_space(search_space=search_space, train_loader=train_loader)

    currentScores = _collect_graph_scores(opt.graph)
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
        f"[VERIFY-LOAD] {('PRUNED' if use_pruned else 'UNPRUNED')} -> {status}. Report: {report_json}"
    )


def _verify_recompute_matches_saved(
    saved_json, dataset, zcp_method, seed, use_pruned, arch_scores_dir, report_json
):
    with open(saved_json, "r") as f:
        payload = json.load(f)
    saved_scores = payload["scores"]

    pre_arch_base = os.path.join(arch_scores_dir, "arch_scores") if use_pruned else None
    config = _build_min_config(
        dataset, seed, zcp_method, pre_op_dir=None, pre_arch_base=pre_arch_base
    )
    opt = PreZCPZCPGSparse(config) if use_pruned else ZCPGSparse(config)

    train_loader, _, _, _, _ = get_train_val_loaders(
        config, train_workers=0, val_workers=0
    )
    search_space = NasBench201SearchSpace(n_classes=_nb201_classes(dataset))
    opt.adapt_search_space(search_space=search_space, train_loader=train_loader)

    currentScores = _collect_graph_scores(opt.graph)
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
        f"[VERIFY-RECOMPUTE] {('PRUNED' if use_pruned else 'UNPRUNED')} -> {'OK' if not mismatches else f'{len(mismatches)} mismatch(es)'}. Report: {report_json}"
    )


def _verify_final_arch_match(
    dataset, zcp_method, seed, use_pruned, pre_dir, arch_scores_dir, report_json
):
    pre_arch_base = os.path.join(arch_scores_dir, "arch_scores") if use_pruned else None
    cfg_loaded = _build_min_config(
        dataset, seed, zcp_method, pre_op_dir=pre_dir, pre_arch_base=pre_arch_base
    )
    cfg_recomp = _build_min_config(
        dataset, seed, zcp_method, pre_op_dir=None, pre_arch_base=pre_arch_base
    )
    Optimizer = PreZCPZCPGSparse if use_pruned else ZCPGSparse

    train_loader, _, _, _, _ = get_train_val_loaders(
        cfg_loaded, train_workers=0, val_workers=0
    )
    n_cls = _nb201_classes(dataset)

    opt_loaded = Optimizer(cfg_loaded)
    opt_loaded.adapt_search_space(
        NasBench201SearchSpace(n_classes=n_cls), train_loader=train_loader
    )

    opt_recomp = Optimizer(cfg_recomp)
    opt_recomp.adapt_search_space(
        NasBench201SearchSpace(n_classes=n_cls), train_loader=train_loader
    )

    final_arch_loaded = opt_loaded.get_final_architecture()
    final_arch_recomp = opt_recomp.get_final_architecture()

    mismatches = []
    if final_arch_loaded.modules_str() != final_arch_recomp.modules_str():
        mismatches.append("Final architecture string representation mismatch.")
        mismatches.append(f"Loaded Arch:\n{final_arch_loaded.modules_str()}")
        mismatches.append(f"Recomputed Arch:\n{final_arch_recomp.modules_str()}")

    os.makedirs(os.path.dirname(report_json), exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(
            {
                "ok": not mismatches,
                "mismatches_count": len(mismatches),
                "details": mismatches,
                "meta": {
                    "dataset": dataset,
                    "zcp_method": zcp_method,
                    "seed": seed,
                    "pruned": use_pruned,
                    "check": "final_arch_match",
                },
            },
            f,
            indent=2,
        )
    print(
        f"[VERIFY-ARCH] {('PRUNED' if use_pruned else 'UNPRUNED')} -> {'OK' if not mismatches else 'FAIL'}. Report: {report_json}"
    )


def main():
    ap = argparse.ArgumentParser("Verify precomputed ZCP scores.")
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
    )
    ap.add_argument(
        "--out_base_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_op_scoring_timefactor",
    )
    ap.add_argument("--skip_unpruned", action="store_true")
    ap.add_argument("--skip_pruned", action="store_true")
    args = ap.parse_args()

    # Set seeds and deterministic settings
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Paths
    unpruned_dir = os.path.join(args.out_base_dir, "unpruned")
    pruned_dir = os.path.join(args.out_base_dir, "pruned")
    base_name = f"op_scores_{args.dataset}_{args.zcp_method}_seed{args.seed}.json"
    unpruned_json = os.path.join(unpruned_dir, base_name)
    pruned_json = os.path.join(pruned_dir, base_name)

    # Run all verifications
    for use_pruned, path, data_dir in [
        (False, unpruned_json, unpruned_dir),
        (True, pruned_json, pruned_dir),
    ]:
        if (use_pruned and args.skip_pruned) or (not use_pruned and args.skip_unpruned):
            continue
        if not os.path.exists(path):
            print(f"[SKIP] Verification skipped, file not found: {path}")
            continue

        print(f"--- Verifying {'Pruned' if use_pruned else 'Unpruned'} Scores ---")

        # Verify loaded vs saved
        report_lvs = os.path.join(
            data_dir,
            f"verify_loaded_vs_saved_{args.dataset}_{args.zcp_method}_seed{args.seed}.json",
        )
        _verify_loaded_matches_saved(
            path,
            args.dataset,
            args.zcp_method,
            args.seed,
            use_pruned,
            data_dir,
            args.arch_scores_dir,
            report_lvs,
        )

        # Verify recomputed vs saved
        report_rvs = os.path.join(
            data_dir,
            f"verify_recompute_vs_saved_{args.dataset}_{args.zcp_method}_seed{args.seed}.json",
        )
        _verify_recompute_matches_saved(
            path,
            args.dataset,
            args.zcp_method,
            args.seed,
            use_pruned,
            args.arch_scores_dir,
            report_rvs,
        )

        # Verify final architecture match
        report_arch = os.path.join(
            data_dir,
            f"verify_final_arch_{args.dataset}_{args.zcp_method}_seed{args.seed}.json",
        )
        _verify_final_arch_match(
            args.dataset,
            args.zcp_method,
            args.seed,
            use_pruned,
            data_dir,
            args.arch_scores_dir,
            report_arch,
        )


if __name__ == "__main__":
    main()
