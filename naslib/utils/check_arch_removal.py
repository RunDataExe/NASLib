import itertools
import numpy as np
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace
from naslib.utils.remove_arch_from_search_space import (
    add_betas_to_edges,
    remove_architecture,
)

EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
OP_NAMES = [
    "Identity",
    "Zero",
    "ReLUConvBN3x3",
    "ReLUConvBN1x1",
    "AvgPool1x1",
]

# 1. Create search space and enumerate all architectures
search_space = NasBench201SearchSpace()
add_betas_to_edges(search_space, scope=search_space.OPTIMIZER_SCOPE)
all_archs = [list(op_indices) for op_indices in search_space.get_arch_iterator()]

# 2. Choose some architectures to remove
# to_remove = [all_archs[0], all_archs[1], all_archs[10]]
to_remove = [
    [0, 1, 2, 3, 4, 0],
    [0, 1, 2, 3, 4, 1],
    [0, 1, 2, 3, 4, 2],
    [0, 1, 2, 3, 4, 3],
    [0, 1, 2, 3, 4, 4],
]

print(f"Removing architectures: {to_remove}")


def print_cell_ops(search_space, msg):
    cell = search_space.edges[2, 3].op  # Get the cell graph
    print(f"\n{msg}")
    for edge in cell.edges:
        ops = cell.edges[edge]["op"]
        op_names = (
            [op.get_op_name for op in ops]
            if isinstance(ops, list)
            else [ops.get_op_name]
        )
        print(f"Edge {edge}: {op_names}")


print_cell_ops(search_space, "Before removal")
# 3. Remove them
for arch in to_remove:
    remove_architecture(search_space, arch, representation_type="op_indices")
print_cell_ops(search_space, "After removal")

# 4. Enumerate remaining architectures
remaining_archs = [list(op_indices) for op_indices in search_space.get_arch_iterator()]

# 5. Check that removed architectures are gone, and others remain
# for arch in to_remove:
# assert arch not in remaining_archs, f"Architecture {arch} was not removed!"

print(f"Removed architectures: {to_remove}")
print(f"Remaining architectures: {len(remaining_archs)}")
# print("Test passed: Only specified architectures were removed.")

# for arch in remaining_archs:
#     try:
#         search_space.set_op_indices(arch)
#         print(f"Architecture {arch} can still be instantiated!")
#     except Exception as e:
#         print(f"Architecture {arch} cannot be instantiated: {e}")


# for arch in to_remove:
#     try:
#         search_space.set_op_indices(arch)
#         print(f"Architecture {arch} can still be instantiated!")
#     except Exception as e:
#         print(f"Architecture {arch} cannot be instantiated: {e}")


def get_valid_arch_iterator(search_space):
    cell = search_space.edges[2, 3].op
    ops_per_edge = [cell.edges[edge]["op"] for edge in EDGE_LIST]
    op_choices = [list(range(len(ops))) for ops in ops_per_edge]
    for indices in itertools.product(*op_choices):
        yield indices


valid_iterator = get_valid_arch_iterator(search_space)
valid_archs = [list(op_indices) for op_indices in valid_iterator]

print(f"Valid architectures: {len(valid_archs)}")


import os
import json
import numpy as np
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace
from naslib.utils.remove_arch_from_search_space import (
    add_betas_to_edges,
    remove_architecture,
)

EDGE_LIST = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
OP_NAMES = [
    "Identity",
    "Zero",
    "ReLUConvBN3x3",
    "ReLUConvBN1x1",
    "AvgPool1x1",
]


def print_cell_ops(search_space, msg):
    cell = search_space.edges[2, 3].op  # Get the cell graph
    print(f"\n{msg}")
    for edge in cell.edges:
        ops = cell.edges[edge]["op"]
        op_names = (
            [op.get_op_name for op in ops]
            if isinstance(ops, list)
            else [ops.get_op_name]
        )
        print(f"Edge {edge}: {op_names}")


def test_remove_worst_archs_with_zc_scores(dataset, zc_scores_prefix):
    # 1. Load precomputed scores
    zc_scores_path = f"{zc_scores_prefix}_{dataset}.json"
    duration_path = zc_scores_path.replace("arch_scores_", "arch_scores_duration_")
    assert os.path.exists(zc_scores_path), f"Missing: {zc_scores_path}"
    assert os.path.exists(duration_path), f"Missing: {duration_path}"

    print(f"\n=== Testing removal for dataset: {dataset} ===")
    with open(zc_scores_path, "r") as f:
        scores = json.load(f)

    op_to_score = {
        tuple(entry["op_indices"]): {
            "jacov": entry["jacov"],
            "synflow": entry["synflow"],
            "params": entry["params"],
        }
        for entry in scores
    }

    # 2. Create search space and enumerate all architectures
    search_space = NasBench201SearchSpace()
    add_betas_to_edges(search_space, scope=search_space.OPTIMIZER_SCOPE)
    arch_list = [list(op_indices) for op_indices in search_space.get_arch_iterator()]
    arch_tuples = [tuple(op) for op in arch_list]

    jacov_scores = []
    synflow_scores = []
    param_scores = []

    for arch in arch_tuples:
        score = op_to_score[arch]
        jacov_scores.append(score["jacov"])
        synflow_scores.append(score["synflow"])
        param_scores.append(score["params"])

    jacov_scores = np.array(jacov_scores)
    synflow_scores = np.array(synflow_scores)
    param_scores = np.array(param_scores)

    # 3. Get indices of worst for each metric
    worst_jacov = np.argsort(jacov_scores)[:750]
    worst_synflow = np.argsort(synflow_scores)[:750]
    worst_params = np.argsort(param_scores)[:750]

    # 4. Find common worst indices across all three metrics
    worst_set = set(worst_jacov) & set(worst_synflow) & set(worst_params)
    print(f"Common worst indices: {sorted(worst_set)}")
    print(f"Number of common worst: {len(worst_set)}")

    print_cell_ops(search_space, "Before removal")
    print("Number of architectures before removal:", len(arch_list))

    # 5. Remove them
    for idx in worst_set:
        remove_architecture(
            search_space, arch_list[idx], representation_type="op_indices"
        )

    # 6. Print after removal
    # remaining_archs = [
    #     list(op_indices) for op_indices in search_space.get_arch_iterator()
    # ]

    print_cell_ops(search_space, "After removal")
    valid_iterator = get_valid_arch_iterator(search_space)
    valid_archs = [list(op_indices) for op_indices in valid_iterator]
    print("Number of valid architectures after removal:", len(valid_archs))


if __name__ == "__main__":
    # Example usage: test for each dataset
    zc_scores_prefix = "naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor/arch_scores"
    for dataset in ["cifar10", "cifar100", "ImageNet16-120"]:
        if os.path.exists(f"{zc_scores_prefix}_{dataset}.json"):
            test_remove_worst_archs_with_zc_scores(dataset, zc_scores_prefix)
