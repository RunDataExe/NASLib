import torch
import logging
from naslib.search_spaces.core.graph import Graph, EdgeData
from naslib.search_spaces.nasbench201.conversions import (
    convert_str_to_op_indices,
    OP_NAMES,
    EDGE_LIST,
)
from naslib.search_spaces.core import primitives as naslib_ops

# OP_NAMES from nasbench201.conversions, e.g.:
# ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool1x1"]
# EDGE_LIST from nasbench201.conversions, e.g.:
# ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))


def add_betas_to_edges(graph: Graph, scope: list = None):
    """
    Adds a 'beta' parameter to operations on edges within the specified scope(s).
    Beta = 1 marks an operation for removal. Call once before removing architectures.

    Args:
        graph (Graph): The search space graph.
        scope (list, optional): List of scope names to apply to.
                                Defaults to graph.OPTIMIZER_SCOPE.
    """
    if scope is None:
        current_scope = (
            graph.OPTIMIZER_SCOPE if hasattr(graph, "OPTIMIZER_SCOPE") else "all"
        )
    else:
        current_scope = scope

    def _add_beta_param_to_edge_data(edge_data: EdgeData):
        if edge_data.has("op") and isinstance(edge_data.op, list):
            num_primitives = len(edge_data.op)
            if num_primitives > 0:  # Only add if there are actual operations
                # Add beta only if it's not there or if its size is inconsistent
                if not edge_data.has("beta") or len(edge_data.beta) != num_primitives:
                    beta = torch.nn.Parameter(
                        torch.zeros(
                            size=[num_primitives], dtype=torch.int8, requires_grad=False
                        ),
                        requires_grad=False,
                    )
                    # 'beta' is private to this edge's list of choices
                    edge_data.set("beta", beta, shared=False)

    graph.update_edges(
        lambda edge: _add_beta_param_to_edge_data(edge.data),  # edge is EdgeAttributes
        scope=current_scope,
        private_edge_data=True,  # beta is private to the list of ops on an edge
    )


def _mark_architecture_betas_on_graph(
    graph: Graph,
    arch_representation,
    scope: list = None,
    representation_type: str = "op_indices",
):
    """Internal helper to mark betas for a given architecture on the graph."""
    if scope is None:
        current_scope = (
            graph.OPTIMIZER_SCOPE if hasattr(graph, "OPTIMIZER_SCOPE") else "all"
        )
    else:
        current_scope = scope

    op_indices_for_arch = []
    if representation_type == "arch_str":
        if not isinstance(arch_representation, str):
            raise ValueError("arch_str representation must be a string.")
        op_indices_for_arch = convert_str_to_op_indices(arch_representation)
    elif representation_type == "op_indices":
        if not (
            isinstance(arch_representation, (list, tuple))
            and all(isinstance(i, int) for i in arch_representation)
        ):
            raise ValueError("op_indices representation must be a list/tuple of ints.")
        op_indices_for_arch = list(arch_representation)  # Ensure it's a list
    else:
        raise ValueError(f"Unsupported representation_type: {representation_type}")

    if len(op_indices_for_arch) != len(EDGE_LIST):
        raise ValueError(
            f"Architecture op_indices length {len(op_indices_for_arch)} "
            f"does not match NB201 EDGE_LIST length {len(EDGE_LIST)}."
        )

    # Map architecture's op choices to cell edges: { (cell_edge_from, cell_edge_to) : op_name_in_arch }
    arch_op_names_on_edges = {}
    for i, op_idx_in_arch in enumerate(op_indices_for_arch):
        edge_tuple_in_cell = EDGE_LIST[i]  # (from_node, to_node) in the cell definition
        if not (0 <= op_idx_in_arch < len(OP_NAMES)):
            raise ValueError(
                f"Op index {op_idx_in_arch} in architecture is out of bounds for OP_NAMES."
            )
        arch_op_names_on_edges[edge_tuple_in_cell] = OP_NAMES[op_idx_in_arch]

    def _mark_beta_on_edge_nb201(edge):
        edge_data = edge.data
        current_edge_tuple_in_cell = (edge.head, edge.tail)
        if current_edge_tuple_in_cell in arch_op_names_on_edges:
            if (
                edge_data.has("beta")
                and edge_data.has("op")
                and isinstance(edge_data.op, list)
            ):
                target_op_name = arch_op_names_on_edges[current_edge_tuple_in_cell]
                # Find the op by name in the current ops list
                for i, op in enumerate(edge_data.op):
                    if op.get_op_name == target_op_name:
                        edge_data.beta[i] = 1
                        break

    # graph.update_edges will apply this to edges of cell subgraphs if scope is correct
    graph.update_edges(
        _mark_beta_on_edge_nb201,
        scope=current_scope,
        private_edge_data=True,  # Modifying beta which is private
    )


def _apply_pruning_to_graph_edges(graph: Graph, scope: list = None):
    """Internal helper to prune operations based on beta flags from graph edges."""
    if scope is None:
        current_scope = (
            graph.OPTIMIZER_SCOPE if hasattr(graph, "OPTIMIZER_SCOPE") else "all"
        )
    else:
        current_scope = scope

    def _prune_ops_from_edge_data(edge_data: EdgeData):
        if (
            edge_data.has("beta")
            and edge_data.has("op")
            and isinstance(edge_data.op, list)
        ):
            original_ops = edge_data.op
            betas = edge_data.beta

            kept_ops = []
            if len(original_ops) != len(betas):
                # Fallback: if mismatch, keep original ops to avoid breaking search space unexpectedly.
                # This state should ideally be prevented by consistent use of add_betas_to_edges.
                # print(f"Warning: Mismatch len(ops)={len(original_ops)} vs len(betas)={len(betas)} on an edge. Skipping pruning for this edge.")
                kept_ops = original_ops
            else:
                for i, op_primitive in enumerate(original_ops):
                    if betas[i] == 1:  # Marked for removal
                        pass
                    else:
                        kept_ops.append(op_primitive)

            if (
                not kept_ops and original_ops
            ):  # All ops were pruned AND there were ops initially
                # print(f"Warning: All ops on an edge were pruned. Adding Zero(stride=1) as fallback.")
                # For NB201 cell, Zero op typically has stride 1.
                # This Zero op should be instantiated correctly if it needs specific parameters (e.g., channels),
                # but naslib_ops.Zero is generally simple.
                kept_ops.append(naslib_ops.Zero(stride=1))

            edge_data.set("op", kept_ops)

            # Reset beta array for the new list of ops, so it's ready for the next removal
            new_beta_param = torch.nn.Parameter(
                torch.zeros(
                    size=[len(kept_ops)], dtype=torch.int8, requires_grad=False
                ),
                requires_grad=False,
            )
            edge_data.set("beta", new_beta_param)

    graph.update_edges(
        lambda edge: _prune_ops_from_edge_data(edge.data),  # edge is EdgeAttributes
        scope=current_scope,
        private_edge_data=True,  # op list and beta are private
    )


def remove_architecture(
    graph: Graph,
    arch_representation,
    scope: list = None,
    representation_type: str = "op_indices",
):
    """
    Removes a given architecture from the search space choices on the specified graph.
    It is assumed that `add_betas_to_edges(graph, scope)` has been called once
    on the graph object before starting to use this function.

    Args:
        graph (Graph): The search space graph (e.g., NasBench201SearchSpace instance).
        arch_representation (str or list/tuple): The architecture to remove.
            If 'op_indices', a list/tuple like (3, 4, 2, 4, 4, 1).
            If 'arch_str', a string like "|nor_conv_1x1~0|+|...|".
        scope (list, optional): List of scope names (e.g., ['stage_1', 'stage_2'])
                                to apply the removal. Defaults to graph.OPTIMIZER_SCOPE.
        representation_type (str, optional): 'op_indices' or 'arch_str'.
                                             Defaults to 'op_indices'.
    """
    logging.debug(
        "Do not use with: set_op_indices, sample_random_architecture (as it calles set_op_indices) convert_op_indices_to_naslib, set_ops, set_cell_ops, check update_edge functions."
    )
    # Step 1: Mark the operations of the specified architecture by setting their beta to 1.
    _mark_architecture_betas_on_graph(
        graph, arch_representation, scope=scope, representation_type=representation_type
    )

    # Step 2: Prune the marked operations from available choices on edges.
    # This function also resets beta flags for the new (reduced) list of operations,
    # making the graph ready for subsequent calls to remove_architecture if needed.
    _apply_pruning_to_graph_edges(graph, scope=scope)
