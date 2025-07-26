import argparse
import json
from fvcore.common.config import CfgNode

import torch
from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace
import os
from naslib.utils import get_project_root, get_train_val_loaders
import time
import numpy as np


def score_and_save_architectures(graph, train_loader, device, save_path):
    """
    Score all architectures, order them, and save to a file.
    """
    start_time = time.time()

    arch_list = [list(op_indices) for op_indices in graph.get_arch_iterator()]
    jacov_pred = ZeroCost(method_type="jacov")
    synflow_pred = ZeroCost(method_type="synflow")
    params_pred = ZeroCost(method_type="params")

    scores = []
    for idx, op_indices in enumerate(arch_list):
        arch_graph = graph.clone()
        arch_graph.set_op_indices(op_indices)
        arch_graph = arch_graph.to(device)
        arch_graph.parse()
        jacov = jacov_pred.query(arch_graph, train_loader)
        synflow = synflow_pred.query(arch_graph, train_loader)
        params = params_pred.query(arch_graph, train_loader)
        print(
            f"Scored architecture {idx + 1}/{len(arch_list)}: "
            f"Jacov: {jacov}, Synflow: {synflow}, Params: {params}"
        )
        scores.append(
            {
                "idx": idx,
                "op_indices": op_indices,
                "jacov": jacov,
                "synflow": synflow,
                "params": params,
            }
        )
    duration = time.time() - start_time
    # save time into json

    duration_save = save_path.replace("arch_scores_", "arch_scores_duration_")

    with open(duration_save, "w") as f:
        json.dump({"duration": duration}, f)
    with open(save_path, "w") as f:
        json.dump(scores, f)
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Run NASLib optimizer with specified configuration."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset (e.g., cifar10, cifar100, ImageNet16-120)",
    )

    args = parser.parse_args()
    seed = 1544457859

    np.random.seed(seed)
    torch.manual_seed(seed)

    config = {
        "data": str(get_project_root()) + "/data",
        "dataset": args.dataset,
        "search": {
            "seed": seed,
            "batch_size": 256,
            "train_portion": 0.8,
            "cutout": False,
        },
    }
    config = CfgNode.load_cfg(json.dumps(config))

    graph = NasBench201SearchSpace()
    train_loader = train_loader, _, _, _, _ = get_train_val_loaders(
        config, train_workers=0, val_workers=0
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_path = f"naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor/arch_scores_{args.dataset}.json"

    scores = score_and_save_architectures(graph, train_loader, device, save_path)
    print(f"Scored {len(scores)} architectures and saved to {save_path}")


if __name__ == "__main__":
    main()
