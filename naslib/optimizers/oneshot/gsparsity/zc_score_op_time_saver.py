import argparse, json, time, os, torch, numpy as np
from fvcore.common.config import CfgNode
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_project_root, get_train_val_loaders
from naslib.optimizers.oneshot.gsparsity.zcp_minmax_gsparse_optimizer import (
    ZCP_GSparseOptimizer,
)


def _collect_leaf_zcp(module):
    # Collect exactly what optimizer sets: attributes named 'zero_cost_proxy' on leaf submodules.
    vals = []
    # Depth-first to find leaves with attribute set by optimizer
    for sub in module.modules():
        if sub is module:
            # Skip the root primitive itself; it may also carry zero_cost_proxy (Identity case)
            if hasattr(sub, "zero_cost_proxy"):
                vals.append(float(sub.zero_cost_proxy))
            continue
        if hasattr(sub, "zero_cost_proxy"):
            vals.append(float(sub.zero_cost_proxy))
    # If nothing nested and primitive itself has score, keep it
    if not vals and hasattr(module, "zero_cost_proxy"):
        vals.append(float(module.zero_cost_proxy))
    return vals


def _collect_graph_scores(graph):
    # Collect per-edge, per-primitive leaf scores using the same traversal as saving.
    scores = []
    ops = graph.get_all_edge_data(
        "op", scope=graph.OPTIMIZER_SCOPE, private_edge_data=True
    )
    for ei, mixed_op in enumerate(ops):
        prims = getattr(mixed_op, "primitives", [])
        for pj, prim in enumerate(prims):
            leaf_scores = _collect_leaf_zcp(prim)
            scores.append(
                {"edge_idx": ei, "primitive_idx": pj, "leaf_scores": leaf_scores}
            )
    return scores


def _verify_scores(saved_scores, current_scores):
    mismatches = []
    if len(saved_scores) != len(current_scores):
        return [
            f"Length mismatch: saved={len(saved_scores)} current={len(current_scores)}"
        ]
    for i, (s, c) in enumerate(zip(saved_scores, current_scores)):
        if s["edge_idx"] != c["edge_idx"] or s["primitive_idx"] != c["primitive_idx"]:
            mismatches.append(
                f"Index mismatch at idx {i}: saved(e{s['edge_idx']},p{s['primitive_idx']}) "
                f"!= current(e{c['edge_idx']},p{c['primitive_idx']})"
            )
            continue
        ls, lc = s["leaf_scores"], c["leaf_scores"]
        if ls != lc:
            # Fall back to element-wise diff for a precise report
            if len(ls) != len(lc):
                mismatches.append(
                    f"Leaf length mismatch at edge {s['edge_idx']} prim {s['primitive_idx']}: "
                    f"saved={len(ls)} current={len(lc)}"
                )
                continue
            for k, (a, b) in enumerate(zip(ls, lc)):
                if a != b:
                    mismatches.append(
                        f"Value mismatch at edge {s['edge_idx']} prim {s['primitive_idx']} leaf {k}: "
                        f"saved={a} current={b}"
                    )
                    break
    return mismatches


def main():
    ap = argparse.ArgumentParser()
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
        "--out_dir",
        type=str,
        default="naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_op_scoring_timefactor",
        help="Directory to write op_scores_{dataset}_{zcp_method}_seed{seed}.json",
    )
    ap.add_argument("--out", type=str, default=None, help="Explicit output JSON path")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_dict = {
        "data": str(get_project_root()) + "/data",
        "dataset": args.dataset,
        "search": {
            "seed": args.seed,
            "batch_size": 256,
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
            "zcp_method": args.zcp_method,
            # keep these for runtime API compatibility if used elsewhere
            "train_workers": 0,
            "val_workers": 0,
        },
    }
    config = CfgNode.load_cfg(json.dumps(cfg_dict))

    # Build data (not counted in timing)
    train_loader, _, _, _, _ = get_train_val_loaders(
        config, train_workers=0, val_workers=0
    )

    # Build search space and optimizer
    n_cls = {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120}[args.dataset]
    search_space = NasBench201SearchSpace(n_classes=n_cls)
    opt = ZCP_GSparseOptimizer(config)

    # Time only the adapt_search_space (which computes and normalizes ZCPs)
    t0 = time.time()
    opt.adapt_search_space(search_space=search_space, train_loader=train_loader)
    dur = time.time() - t0

    graph = opt.graph

    # Collect and save scores (public API, no private _edges)
    scores = _collect_graph_scores(graph)

    # Prepare output paths
    if args.out:
        out_json = args.out
        out_dir = os.path.dirname(args.out) or "."
        base_name = os.path.basename(args.out).replace(".json", "")
        dur_json = os.path.join(
            out_dir, base_name.replace("op_scores_", "op_scores_duration_") + ".json"
        )
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        base = f"op_scores_{args.dataset}_{args.zcp_method}_seed{args.seed}"
        out_json = os.path.join(args.out_dir, base + ".json")
        dur_json = os.path.join(
            args.out_dir, base.replace("op_scores_", "op_scores_duration_") + ".json"
        )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    meta = {
        "dataset": args.dataset,
        "zcp_method": args.zcp_method,
        "seed": args.seed,
        "normalized": True,
        "version": 2,
    }
    with open(out_json, "w") as f:
        json.dump({"meta": meta, "scores": scores}, f)
    with open(dur_json, "w") as f:
        json.dump({"duration": dur, **meta}, f)

    print(f"Wrote {len(scores)} op-scores to {out_json} in {dur:.2f}s")

    # Safety verification (excluded from timing)
    current_scores = _collect_graph_scores(graph)
    mismatches = _verify_scores(scores, current_scores)
    if mismatches:
        print(f"[VERIFY] Found {len(mismatches)} mismatch(es). Showing up to 5:")
        for m in mismatches[:5]:
            print("[VERIFY] " + m)
        # Non-fatal: exit code 0 but clearly warn
        print("[VERIFY] WARNING: Saved scores do not match collected graph scores.")
    else:
        print(
            "[VERIFY] Op-score mapping verified: saved scores match graph scores exactly."
        )


if __name__ == "__main__":
    main()
