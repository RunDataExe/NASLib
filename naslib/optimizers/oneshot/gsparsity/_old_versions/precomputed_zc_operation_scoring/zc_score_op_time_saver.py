import argparse
import json
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

# --- HELPER FUNCTIONS (RETAINED FOR SAVING) ---


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


def _nb201_classes(dataset):
    return {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}[dataset]


def _build_min_config(dataset, seed, zcp_method, pre_arch_base=None):
    cfg = {
        "data": str(get_project_root()) + "/data",
        "dataset": dataset,
        "search": {
            "seed": seed,
            "batch_size": 64,
            "train_portion": 0.8,
            "cutout": False,
            "zcp_method": zcp_method,
            "grad_clip": None,
            "weight_decay": 60.0,
            "threshold": 1e-6,
            "normalization": "div",
            "normalization_exponent": 0.5,
            "learning_rate": 1e-3,
            "momentum": 0.8,
        },
    }
    if pre_arch_base:
        cfg["search"]["pre_computed_zc_scores"] = pre_arch_base
    return CfgNode.load_cfg(json.dumps(cfg))


def _run_precompute_once(
    dataset, zcp_method, seed, out_json, use_pruned, arch_scores_dir
):
    pre_arch_base = os.path.join(arch_scores_dir, "arch_scores") if use_pruned else None
    config = _build_min_config(dataset, seed, zcp_method, pre_arch_base=pre_arch_base)

    train_loader, _, _, _, _ = get_train_val_loaders(
        config, train_workers=0, val_workers=0
    )
    search_space = NasBench201SearchSpace(n_classes=_nb201_classes(dataset))

    opt = PreZCPZCPGSparse(config) if use_pruned else ZCPGSparse(config)
    opt.adapt_search_space(search_space=search_space, train_loader=train_loader)

    saved_scores = _collect_graph_scores(opt.graph)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "meta": {
                    "dataset": dataset,
                    "zcp_method": zcp_method,
                    "seed": seed,
                    "normalized": True,
                    "pruned": use_pruned,
                },
                "scores": saved_scores,
            },
            f,
        )

    duration = float(getattr(opt, "_zcp_compute_time", 0.0))
    dur_path = out_json.replace("op_scores_", "op_scores_duration_")
    with open(dur_path, "w") as f:
        json.dump({"duration": duration}, f)

    print(f"[SAVE] Scores for {'pruned' if use_pruned else 'unpruned'} -> {out_json}")
    print(f"[TIME] Saved op precompute duration {duration:.3f}s -> {dur_path}")


def main():
    ap = argparse.ArgumentParser("Precompute per-op ZCPs (unpruned/pruned).")
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

    if not args.skip_unpruned:
        _run_precompute_once(
            args.dataset,
            args.zcp_method,
            args.seed,
            os.path.join(unpruned_dir, base_name),
            False,
            args.arch_scores_dir,
        )

    if not args.skip_pruned:
        _run_precompute_once(
            args.dataset,
            args.zcp_method,
            args.seed,
            os.path.join(pruned_dir, base_name),
            True,
            args.arch_scores_dir,
        )


if __name__ == "__main__":
    main()
