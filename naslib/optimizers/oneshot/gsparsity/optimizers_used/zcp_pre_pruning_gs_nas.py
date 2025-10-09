from dataclasses import replace
from distutils.command.config import config
from locale import normalize
import logging
import os
from random import random
from turtle import pos, position
from matplotlib.colors import NoNorm
import torch.nn.utils.parametrize as P
import torch
from collections.abc import Iterable
from naslib.utils import SimpleStateDict
import json
from naslib.search_spaces.core.primitives import MixedOp
from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.utils import count_parameters_in_MB
from naslib.search_spaces.core.query_metrics import Metric

from naslib.optimizers.oneshot.gsparsity.ProxSGD_for_groups import ProxSGD
import naslib.search_spaces.core.primitives as primitives
from naslib.utils.shape_annotator import ShapeAnnotator

import math
import numpy as np
import os, json

from naslib.predictors.zerocost import ZeroCost
from naslib.utils.remove_arch_from_search_space import (
    add_betas_to_edges,
    remove_architecture,
)
from naslib.predictors.zerocost import ZeroCost
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace


from naslib.optimizers.oneshot.gsparsity.operation_zero_cost_proxy_scoring import (
    evaluate_micro_architecture_zcp,
)

logger = logging.getLogger(__name__)


class ZCP_GSparseOptimizer(MetaOptimizer):
    """
    Implements Group Sparsity as defined in
        Chatzimichailidis et. al. :
        GSparsity: Unifying Network Pruning and
        Neural Architecture Search by Group Sparsity
    """

    mu = 0
    using_step_function = True

    def __init__(
        self,
        config,
        op_optimizer: torch.optim.Optimizer = ProxSGD,
        op_optimizer_evaluate: torch.optim.Optimizer = torch.optim.SGD,
        loss_criteria=torch.nn.CrossEntropyLoss(),
    ):
        """
        Instantiate the optimizer

        Group sparsity paper uses ProxSGD for optimizing operation weights
            during search phase.
        And SGD for optimizing weights during evaluation phase.

        Args:
            epochs (int): Number of epochs. Required for tau
            mu (float): corresponds to the Weight decay
            threshold (float): threshold of pruning
            op_optimizer (torch.optim.Optimizer: ProxSGD): optimizer for the op weights
            op_optmizer_evaluate: (torch.optim.Optimizer): optimizer for the op weights
            loss_criteria: The loss.
            grad_clip (float): Clipping of the gradients. Default None.
        """
        super(ZCP_GSparseOptimizer, self).__init__()

        self.config = config
        self.op_optimizer = op_optimizer
        self.op_optimizer_evaluate = op_optimizer_evaluate
        self.loss = loss_criteria
        self.dataset = config.dataset
        self.grad_clip = config.search.grad_clip
        self.mu = config.search.weight_decay
        self.threshold = config.search.threshold
        self.normalization = config.search.normalization
        self.normalization_exponent = config.search.normalization_exponent
        self.operation_weights = torch.nn.ParameterList()
        self.zcp_method = config.search.zcp_method
        self.train_loader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pruned_op_indices = []  # List of op_indices to prune
        self.best_arch = None

        self.enable_pre_zc_pruning = True
        self.enable_zcp_scaling = False

    @staticmethod
    def check_pruning_invariants(optimizer, graph, scope):
        # 1) collect all optimizer params
        opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}

        n_edges = n_pruned = n_active = n_bad = 0

        def _check(edge):
            nonlocal n_edges, n_pruned, n_active, n_bad
            if edge.data.has("alpha") and hasattr(edge.data.op, "primitives"):
                n_edges += 1
                for i, prim in enumerate(edge.data.op.primitives):
                    is_pruned = bool(getattr(prim, "was_pruned_slot", False))
                    has_params = any(id(p) in opt_param_ids for p in prim.parameters())
                    if is_pruned:
                        n_pruned += 1
                        if has_params:  # pruned slot must not carry trainable params
                            n_bad += 1
                    else:
                        n_active += 1

        graph.update_edges(_check, scope=scope, private_edge_data=True)
        logger.info(
            f"[check] edges={n_edges} active_prims={n_active} pruned_placeholders={n_pruned} bad_pruned_with_params={n_bad}"
        )
        assert n_bad == 0, (
            "Found pruned placeholders leaking parameters into optimizer!"
        )

    @staticmethod
    def dump_edge_prims(graph, scope):
        counts = []

        def _dump(edge):
            if hasattr(edge.data.op, "primitives"):
                names = [
                    (i, type(p).__name__, bool(getattr(p, "was_pruned_slot", False)))
                    for i, p in enumerate(edge.data.op.primitives)
                ]
                counts.append((f"{edge.head}->{edge.tail}", names))

        graph.update_edges(_dump, scope=scope, private_edge_data=True)
        for e, names in counts[:12]:
            logger.info(f"{e} : {names}")
        logger.info(f"edges listed: {len(counts)}")

    def _load_and_apply_precomputed_op_scores(self, graph, scope) -> bool:
        op_dir = getattr(self.config.search, "pre_computed_op_zc_scores_dir", None)
        if not op_dir or not getattr(self.config.search, "zcp_method", None):
            return False

        fname = os.path.join(
            op_dir,
            f"op_scores_{self.dataset}_{self.zcp_method}_seed{self.config.search.seed}.json",
        )
        if not os.path.exists(fname):
            logger.info(f"Precomputed op ZCP file not found: {fname}")
            return False

        with open(fname, "r") as f:
            payload = json.load(f)

        meta = payload.get("meta", {})
        scores = payload.get("scores", [])
        if not isinstance(scores, list) or not scores:
            logger.warning(f"Precomputed op ZCP file malformed or empty: {fname}")
            return False

        normalized = bool(meta.get("normalized", False))

        # Traverse edges/primitives exactly as used by weight accumulation
        ops = graph.get_all_edge_data("op", scope=scope, private_edge_data=True)
        expected = sum(len(getattr(m, "primitives", [])) for m in ops)
        if len(scores) != expected:
            logger.warning(
                "Precomputed op ZCP shape mismatch. Falling back to on-the-fly computation."
            )
            return False

        idx = 0
        for mixed_op in ops:
            prims = getattr(mixed_op, "primitives", [])
            for prim in prims:
                entry = scores[idx]
                idx += 1

                leaf_scores = list(entry.get("leaf_scores", []) or [])
                prim_score = entry.get("primitive_score", None)

                # Assign to j-level leaves first; then to k-level if leftovers exist
                try:
                    J = len(prim.op)
                    # j-level
                    for j in range(J):
                        if leaf_scores:
                            try:
                                prim.op[j].zero_cost_proxy = float(leaf_scores.pop(0))
                            except Exception as e:
                                logger.info("Failed assigning j-level ZCP: %s", e)
                    # k-level (if any remain)
                    if leaf_scores:
                        for j in range(J):
                            try:
                                K = len(prim.op[j].op)
                                for k in range(K):
                                    if leaf_scores:
                                        prim.op[j].op[k].zero_cost_proxy = float(
                                            leaf_scores.pop(0)
                                        )
                            except AttributeError:
                                continue
                except AttributeError:
                    # Simple primitive without children
                    if leaf_scores:
                        prim.zero_cost_proxy = float(leaf_scores.pop(0))

                # Set primitive-level score if present
                if prim_score is not None:
                    try:
                        prim.zero_cost_proxy = float(prim_score)
                    except Exception as e:
                        logger.info("Failed assigning primitive-level ZCP: %s", e)

                if leaf_scores:
                    logger.warning(
                        "Unconsumed leaf_scores remain for a primitive; JSON and graph structure may differ."
                    )

        logger.info(
            "Loaded and applied precomputed per-op ZCPs from %s (normalized=%s)",
            fname,
            normalized,
        )
        self._zcp_compute_time = 0.0
        return True

    def _calculate_and_set_zcp_scores(self, graph, scope):
        """
        Helper function to calculate ZCP scores for all operations once and store them.
        This should only be called once.
        """
        if self._load_and_apply_precomputed_op_scores(graph, scope):
            logger.info(
                "Applied precomputed per-operation ZCP scores (already normalized)."
            )
            return

        logger.info(
            "Calculating all ZCP scores once (method=%s, dataset=%s, scope=%s).",
            self.zcp_method,
            self.dataset,
            str(scope),
        )
        raw_zero_cost_proxy_scores = []
        scored_counter = {"n": 0}

        def collect_zero_cost_proxy_scores(edge):
            if edge.data.has("alpha"):
                for i, prim in enumerate(edge.data.op.primitives):
                    # Skip pruned slots (placeholders)
                    if getattr(prim, "was_pruned_slot", False):
                        logger.debug(
                            "Skip ZCP scoring pruned slot edge(%s->%s) prim=%d %s",
                            edge.head,
                            edge.tail,
                            i,
                            type(prim).__name__,
                        )
                        continue
                    try:
                        input_shape = prim.shapes["input_shape"]
                        output_shape = prim.shapes["output_shape"]
                    except (AttributeError, KeyError, TypeError):
                        logger.info(
                            "Skip scoring edge(%s->%s) prim=%d (missing shapes).",
                            edge.head,
                            edge.tail,
                            i,
                        )
                        continue
                    try:
                        score = evaluate_micro_architecture_zcp(
                            operation=prim,
                            operation_input_full_shape=input_shape,
                            operation_output_full_shape=output_shape,
                            dataloader=self.train_loader,
                            zcp_method=self.zcp_method,
                            dataset=self.dataset,
                        )
                    except Exception as exc:
                        logger.info(
                            "Failed scoring edge(%s->%s) prim=%d (%s).",
                            edge.head,
                            edge.tail,
                            i,
                            exc,
                        )
                        continue
                    prim.zero_cost_proxy = score
                    raw_zero_cost_proxy_scores.append(score)
                    scored_counter["n"] += 1
                    logger.info(
                        "ZCP raw score edge(%s->%s) prim=%d %s: %.6f",
                        edge.head,
                        edge.tail,
                        i,
                        type(prim).__name__,
                        float(score),
                    )

        graph.update_edges(
            collect_zero_cost_proxy_scores, scope=scope, private_edge_data=True
        )
        logger.info("Computed raw ZCP scores for %d ops.", scored_counter["n"])

        if not raw_zero_cost_proxy_scores:
            logger.warning("ZCP scores list is empty. Skipping normalization.")
            return

        min_zcp, max_zcp = (
            min(raw_zero_cost_proxy_scores),
            max(raw_zero_cost_proxy_scores),
        )
        logger.info(
            "Raw ZCP scores range: [%.6f, %.6f]", float(min_zcp), float(max_zcp)
        )

        # Define normalization function based on method
        if self.zcp_method == "params":

            def normalize(raw_score, min_val=min_zcp, max_val=max_zcp):
                if max_val == min_val:
                    return 1.0
                return (np.log(raw_score + 1e-9) - np.log(min_val + 1e-9)) / (
                    np.log(max_val + 1e-9) - np.log(min_val + 1e-9)
                )

            logger.info("Using 'params' log-normalization for ZCP.")
        elif self.zcp_method == "synflow":

            def normalize(raw_score, min_val=min_zcp, max_val=max_zcp):
                if max_val == min_val:
                    return 1.0
                shift = abs(min_val) + 1e-9
                return (np.log(raw_score + shift) - np.log(min_val + shift)) / (
                    np.log(max_val + shift) - np.log(min_val + shift)
                )

            logger.info("Using 'synflow' shifted log-normalization for ZCP.")
        else:

            def normalize(raw_score, min_val=min_zcp, max_val=max_zcp):
                if max_val == min_val:
                    return 1.0
                return (raw_score - min_val) / (max_val - min_val)

            logger.info("Using linear normalization for ZCP.")

        normalized_counter = {"n": 0}

        def normalize_and_apply_scores(edge):
            if edge.data.has("alpha"):
                for i, prim in enumerate(edge.data.op.primitives):
                    # Skip pruned slots
                    if getattr(prim, "was_pruned_slot", False):
                        continue
                    if not hasattr(prim, "zero_cost_proxy"):
                        continue
                    before = float(prim.zero_cost_proxy)
                    prim.zero_cost_proxy = normalize(before)
                    normalized_counter["n"] += 1
                    logger.info(
                        "ZCP normalized edge(%s->%s) prim=%d %s: %.6f -> %.6f",
                        edge.head,
                        edge.tail,
                        i,
                        type(prim).__name__,
                        before,
                        float(prim.zero_cost_proxy),
                    )

        graph.update_edges(
            normalize_and_apply_scores, scope=scope, private_edge_data=True
        )
        logger.info(
            "Completed ZCP normalization. Applied to %d ops.", normalized_counter["n"]
        )

    @staticmethod
    def update_ops(edge):
        """
        Function to replace the primitive ops at the edges
        with the GSparse specific GSparseMixedOp.
        """
        primitives = edge.data.op
        edge.data.set("op", GSparseMixedOp(primitives))

    @staticmethod
    def add_alphas(edge):
        """
        Function to add the pruning flag 'alpha' to the edges.
        And add a parameter 'weights' that will be used for storing the l2 norm
        of the weights of the operations which later is used for pruning.
        """
        len_primitives = len(edge.data.op)
        alpha = torch.nn.Parameter(
            torch.zeros(size=[len_primitives], requires_grad=False), requires_grad=False
        )
        weights = torch.nn.Parameter(
            torch.FloatTensor(len_primitives * [0.0]), requires_grad=False
        )
        dimension = torch.nn.Parameter(
            torch.FloatTensor(len_primitives * [0.0]), requires_grad=False
        )
        zero_cost_proxy = torch.nn.Parameter(
            torch.FloatTensor(len_primitives * [0.0]), requires_grad=False
        )
        edge.data.set("alpha", alpha, shared=True)
        edge.data.set("weights", weights, shared=True)
        edge.data.set("dimension", dimension, shared=True)
        edge.data.set("zero_cost_proxy", zero_cost_proxy, shared=True)

    @staticmethod
    def add_weights(edge):
        """
        Operations like Identity(), Zero(stride=1) etc do not have weights of their own,
        neither contained suboperations that have weights, just to optimize over such operations
        we attach a 'weight' parameter to them, which is used in the forward() of the MixedOp
        thus updating them and optimizing over them.
        IMPORTANT: In GroupSparsity, suboperation that do not have weights are ignored from
        optimization point of view, i.e. they are not given weights explicitely to be used while
        calculating the weight of the operation containing them.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(len(edge.data.op)):
            try:
                len(edge.data.op[i].op)
            except AttributeError:
                # Skip attaching weights to pruned placeholders to keep them inert
                if getattr(edge.data.op[i], "was_pruned_slot", False):
                    continue
                weight = torch.nn.Parameter(
                    torch.FloatTensor([1.0]), requires_grad=True
                )
                edge.data.op[i].register_parameter("weight", weight)

    # -------------------------
    # Grouping and ZCP helpers
    # -------------------------
    def _iter_edge_primitives(self, graph, scope):
        """
        Collect (edge, prim_index, primitive_module) for all edges in scope that have alpha.
        Use update_edges to traverse nested graphs with correct scope handling.
        """
        items = []

        def _collect(edge):
            if edge.data.has("alpha") and hasattr(edge.data.op, "primitives"):
                n_prims = len(edge.data.op.primitives)
                logger.info(
                    "Visit edge(%s->%s): %d primitives.", edge.head, edge.tail, n_prims
                )
                for i, prim in enumerate(edge.data.op.primitives):
                    # Skip pruned slots
                    if getattr(prim, "was_pruned_slot", False):
                        continue
                    items.append((edge, i, prim))
                    logger.info(
                        "Edge (%s->%s) prim=%d %s actually collected.",
                        edge.head,
                        edge.tail,
                        i,
                        type(prim).__name__,
                    )

        graph.update_edges(_collect, scope=scope, private_edge_data=True)
        logger.info(
            "Collected %d (edge, prim) pairs in scope %s.", len(items), str(scope)
        )
        return items

    def _group_key(self, edge, prim_idx):
        """
        Group primitives per edge: each (u,v, prim_idx) is its own group.
        """
        key = f"edge({edge.head},{edge.tail})_prim{prim_idx}"
        # Very frequent; keep at debug if needed
        logger.info("Group key resolved: %s", key)
        return key

    def _collect_leaf_modules_with_zcp(self, prim):
        """
        Collect all leaf modules under a primitive that have 'zero_cost_proxy' set.
        """
        leaves = []

        def visit(m):
            # If child has children, traverse; else it's a leaf
            children = list(m.children())
            if len(children) == 0:
                if hasattr(m, "zero_cost_proxy"):
                    leaves.append(m)
            else:
                for c in children:
                    visit(c)

        # Include the primitive itself if it carries a score
        if hasattr(prim, "zero_cost_proxy"):
            leaves.append(prim)
        # Traverse children
        for c in prim.children():
            visit(c)

        logger.info(
            "Primitive %s: collected %d leaf ZCP modules.",
            type(prim).__name__,
            len(leaves),
        )
        return leaves

    def _avg_primitive_zcp(self, prim, default=0):
        """
        Return the primitive-level normalized ZCP score.
        """
        val = float(getattr(prim, "zero_cost_proxy", default))
        logger.info("Primitive %s: ZCP=%.6f.", type(prim).__name__, val)
        return val

    def _primitive_params(self, prim):
        """
        All torch Parameters belonging to this primitive (including nested).
        For parameter-less ops where you attached 'weight', this is included too.
        """
        params = list(prim.parameters())
        n_tensors = len(params)
        n_elems = int(sum(p.numel() for p in params))
        logger.info(
            "Primitive %s: %d parameter tensors (%d elements) collected.",
            type(prim).__name__,
            n_tensors,
            n_elems,
        )
        return params

    def _build_param_groups_with_zcp(
        self,
        graph,
        scope,
        base_mu,
        enable_zcp_scaling=True,
    ):
        """
        Create optimizer param_groups with group-specific weight_decay scaled by ZCP.
        If enable_zcp_scaling is False, grouping remains identical but weight_decay is uniform (= base_mu).
        """
        groups = {}
        for edge, i, prim in self._iter_edge_primitives(graph, scope):
            key = self._group_key(edge, i)
            if key not in groups:
                groups[key] = {"params": [], "zcp_vals": []}
            p = self._primitive_params(prim)
            if enable_zcp_scaling:
                z = self._avg_primitive_zcp(prim, default=0)
                groups[key]["zcp_vals"].append(z)
            groups[key]["params"].extend(p)

        total_groups = len(groups)
        total_params = sum(len(v["params"]) for v in groups.values())
        total_zcp_vals = sum(len(v["zcp_vals"]) for v in groups.values())
        logger.info(
            "[init groups] groups=%d, total_params=%d, total_zcp_vals=%d (scaling=%s)",
            total_groups,
            total_params,
            total_zcp_vals,
            str(enable_zcp_scaling),
        )

        group_scores = {}
        if enable_zcp_scaling:
            for k, v in groups.items():
                zcps = v["zcp_vals"]
                group_scores[k] = float(np.median(zcps)) if zcps else 0.0
                logger.info(
                    "Group %s: median aggregated ZCP=%.6f from %d values.",
                    k,
                    group_scores[k],
                    len(zcps),
                )
            scales = {k: max(1.0 - s, 1e-9) for k, s in group_scores.items()}
        else:
            # uniform scaling
            scales = {k: 1.0 for k in groups.keys()}
            group_scores = {k: None for k in groups.keys()}
            logger.info(
                "ZCP scaling disabled: using uniform weight_decay=base_mu for all groups."
            )

        param_groups = []
        for k, v in groups.items():
            if not v["params"]:
                continue
            wd = float(base_mu) * float(scales[k])
            pg = {
                "params": v["params"],
                "weight_decay": wd,
                "zcp_scale": float(scales[k]),
                "group_key": k,
                "lr": self.config.search.learning_rate,
                "momentum": self.config.search.momentum,
                "clip_bounds": (0, 1),
                "normalization": self.normalization,
                "normalization_exponent": self.normalization_exponent,
                "n_params": len(v["params"]),
                "n_zcp": len(v["zcp_vals"]),
            }
            logger.info(
                "Param group %s: n_params=%d, median_zcp=%s, scale=%.6f, weight_decay=%.6f",
                k,
                len(v["params"]),
                "NA" if group_scores[k] is None else f"{group_scores[k]:.6f}",
                scales[k],
                wd,
            )
            param_groups.append(pg)

        logger.info(
            "Finalized %d param groups (scaling=%s).",
            len(param_groups),
            str(enable_zcp_scaling),
        )
        return param_groups

    def _refresh_group_weight_decay_from_zcp(
        self,
        optimizer,
        graph,
        scope,
        base_mu,
    ):
        """
        Recompute per-group ZCP scales and update optimizer.param_groups in-place.
        If scaling disabled, simply reset all group weight_decay to base_mu.
        """
        if not self.enable_zcp_scaling:
            for pg in optimizer.param_groups:
                pg["weight_decay"] = float(base_mu)
                pg["zcp_scale"] = 1.0
            logger.info(
                "ZCP scaling disabled: refreshed all groups to uniform weight_decay=%.6f.",
                float(base_mu),
            )
            return
        logger.info(
            "Refreshing group weight_decay from ZCP (base_mu=%.6f).", float(base_mu)
        )
        self._calculate_and_set_zcp_scores(graph, scope)

        # Re-aggregate per-group averages
        groups = {}
        for edge, i, prim in self._iter_edge_primitives(graph, scope):
            key = self._group_key(edge, i)
            groups.setdefault(key, []).append(self._avg_primitive_zcp(prim, default=0))

        # Summary before applying updates
        total_groups = len(groups)
        total_zcp_vals = sum(len(v) for v in groups.values())
        avg_zcp_per_group = total_zcp_vals / total_groups if total_groups else 0.0
        logger.info(
            "[refresh groups] groups=%d, total_zcp_vals=%d, avg_zcp/group=%.2f",
            total_groups,
            total_zcp_vals,
            avg_zcp_per_group,
        )

        updated = 0
        total_params = 0
        # Update in-place (scale = 1 - median zcp)
        for pg in optimizer.param_groups:
            key = pg.get("group_key", None)
            if key is None or key not in groups:
                logger.info("Skip optimizer group without matching key: %s", str(key))
                continue
            gscore = float(np.median(groups[key])) if groups[key] else 0.0
            scale = max(1.0 - gscore, 1e-9)
            pg["weight_decay"] = float(base_mu) * float(scale)
            pg["zcp_scale"] = float(scale)

            # update and log counts
            n_params = len(pg.get("params", []))
            n_zcp = len(groups[key])
            pg["n_params"] = n_params
            pg["n_zcp"] = n_zcp
            total_params += n_params

            updated += 1
            logger.info(
                "Updated group %s: n_params=%d, n_zcp=%d, median_zcp=%.6f, "
                "scale=%.6f, weight_decay=%.6f",
                key,
                n_params,
                n_zcp,
                gscore,
                scale,
                float(pg["weight_decay"]),
            )

        logger.info(
            "Refreshed weight_decay for %d optimizer groups (total_params=%d).",
            updated,
            total_params,
        )

    def adapt_search_space(
        self,
        search_space,
        scope=None,
        resume_from_path=None,
        train_loader=None,
        dataset_api=None,
        **kwargs,
    ):
        """
        Modify the search space to fit the optimizer's needs,
        including pre-reducing the search space using zero-cost pruning.
        """
        self.search_space = search_space
        self.train_loader = train_loader
        self.dataset_api = dataset_api

        # Work on a single graph instance end-to-end.
        graph = search_space.clone()

        # If there is no scope defined, use the search space default one
        if not scope:
            scope = graph.OPTIMIZER_SCOPE

        # ------------------------------------------------------------------
        # 0) PRE-REDUCTION PHASE (architecture-level ZC pruning)
        # ------------------------------------------------------------------
        add_betas_to_edges(graph, scope=scope)
        logger.info("Added beta parameters to search space edges.")

        # Always reapply pruning from checkpoint if resuming (shape consistency),
        # otherwise gate pruning by the new flag.
        if resume_from_path and os.path.exists(resume_from_path):
            checkpoint = torch.load(resume_from_path, map_location="cpu")
            logger.debug("Checkpoint keys: %s", list(checkpoint.keys()))
            if "pruned_op_indices" in checkpoint:
                state = checkpoint["pruned_op_indices"]
                if hasattr(state, "state_dict"):
                    self.pruned_op_indices = state.state_dict()["pruned_op_indices"]
                elif isinstance(state, dict):
                    self.pruned_op_indices = state["pruned_op_indices"]
                logger.info(
                    "Loaded %d pruned architectures from checkpoint.",
                    len(self.pruned_op_indices),
                )
                for op_indices in self.pruned_op_indices:
                    remove_architecture(
                        graph,
                        op_indices,
                        scope=scope,
                        representation_type="op_indices",
                    )

        # Option B: use pre-computed zc scores to prune
        if self.enable_pre_zc_pruning:
            logger.info("Pre ZC pruning enabled by config flag.")
            if (
                hasattr(self.config.search, "pre_computed_zc_scores")
                and getattr(self.config.search, "pre_computed_zc_scores")
                and os.path.exists(
                    getattr(self.config.search, "pre_computed_zc_scores")
                    + "_"
                    + self.config.dataset
                    + ".json"
                )
            ):
                pre_computed_zc_scores_path = (
                    getattr(self.config.search, "pre_computed_zc_scores")
                    + "_"
                    + self.config.dataset
                    + ".json"
                )
                duration_path = pre_computed_zc_scores_path.replace(
                    "arch_scores_", "arch_scores_duration_"
                )
                if os.path.exists(duration_path):
                    logger.info(
                        f"Loading zero-cost scores from {getattr(self.config.search, 'pre_computed_zc_scores')}"
                    )
                    with open(pre_computed_zc_scores_path, "r") as f:
                        scores = json.load(f)

                    op_to_score = {
                        tuple(entry["op_indices"]): {
                            "jacov": entry["jacov"],
                            "synflow": entry["synflow"],
                            "params": entry["params"],
                        }
                        for entry in scores
                    }

                    arch_list = [
                        list(op_indices) for op_indices in graph.get_arch_iterator()
                    ]
                    arch_tuples = [tuple(op) for op in arch_list]
                    jacov_scores = np.array(
                        [op_to_score[a]["jacov"] for a in arch_tuples]
                    )
                    synflow_scores = np.array(
                        [op_to_score[a]["synflow"] for a in arch_tuples]
                    )
                    param_scores = np.array(
                        [op_to_score[a]["params"] for a in arch_tuples]
                    )

                    worst_jacov = np.argsort(jacov_scores)[:750]
                    worst_synflow = np.argsort(synflow_scores)[:750]
                    worst_params = np.argsort(param_scores)[:750]
                    worst_set = (
                        set(worst_jacov) & set(worst_synflow) & set(worst_params)
                    )

                    self.pruned_op_indices = [arch_list[idx] for idx in worst_set]
                    for idx in worst_set:
                        remove_architecture(
                            graph,
                            arch_list[idx],
                            scope=scope,
                            representation_type="op_indices",
                        )
                    logger.info(
                        "Removed %d architectures from the search space.",
                        len(worst_set),
                    )

            # Option C: on-the-fly scoring and pruning (full space)
            else:
                logger.info(
                    "No resume path or pre-computed ZC scores found. Pruning from full search space."
                )
                arch_list = [
                    list(op_indices) for op_indices in graph.get_arch_iterator()
                ]
                logger.info(
                    "Enumerating %d architectures (full space).", len(arch_list)
                )

                jacov_pred = ZeroCost(method_type="jacov")
                synflow_pred = ZeroCost(method_type="synflow")
                params_pred = ZeroCost(method_type="params")

                jacov_scores, synflow_scores, param_scores = [], [], []
                for idx, op_indices in enumerate(arch_list):
                    g = search_space.clone()
                    g.set_op_indices(op_indices)
                    g = g.to(self.device)
                    g.parse()
                    jacov_scores.append(jacov_pred.query(g, self.train_loader))
                    synflow_scores.append(synflow_pred.query(g, self.train_loader))
                    param_scores.append(params_pred.query(g, self.train_loader))

                worst_jacov = np.argsort(jacov_scores)[:750]
                worst_synflow = np.argsort(synflow_scores)[:750]
                worst_params = np.argsort(param_scores)[:750]
                worst_set = set(worst_jacov) & set(worst_synflow) & set(worst_params)

                self.pruned_op_indices = [arch_list[idx] for idx in worst_set]
                for idx in worst_set:
                    remove_architecture(
                        graph,
                        arch_list[idx],
                        scope=scope,
                        representation_type="op_indices",
                    )
                logger.info(
                    "Removed %d architectures from the search space.", len(worst_set)
                )
        else:
            logger.info("Pre ZC pruning disabled by config flag.")

        # ------------------------------------------------------------------
        # 1) NOW add alpha/weights sized to the PRUNED op lists
        # ------------------------------------------------------------------
        graph.update_edges(
            self.__class__.add_alphas, scope=scope, private_edge_data=False
        )
        graph.update_edges(
            self.__class__.add_weights, scope=scope, private_edge_data=True
        )

        # 2) Replace primitive lists with GSparseMixedOp
        graph.update_edges(
            self.__class__.update_ops, scope=scope, private_edge_data=True
        )

        # Keep a handle on the per-edge weights tensors (for logging later)
        for alpha in graph.get_all_edge_data("weights"):
            self.operation_weights.append(alpha)

        graph.parse()
        print(graph)

        # 3) Annotate shapes for ZCP scoring
        shape_annotator = ShapeAnnotator(self.config)
        graph = shape_annotator.annotate_graph(graph)

        # 4) Assign to self and continue as usual
        self.graph = graph
        self.scope = scope

        # Compute ZCP, build param-groups and optimizer
        if self.enable_zcp_scaling:
            self._calculate_and_set_zcp_scores(self.graph, self.scope)
        else:
            logger.info("ZCP scoring & scaling disabled: skipping score computation.")
        param_groups = self._build_param_groups_with_zcp(
            self.graph,
            self.scope,
            base_mu=self.mu,
            enable_zcp_scaling=self.enable_zcp_scaling,
        )

        if len(param_groups) == 0:
            raise RuntimeError(
                "No parameter groups found after pre-pruning. Check pruning thresholds and edge attributes."
            )

        self.op_optimizer = self.op_optimizer(
            param_groups,
            lr=self.config.search.learning_rate,
            momentum=self.config.search.momentum,
            weight_decay=None,  # per-group values provided
            clip_bounds=(0, 1),
            normalization=self.normalization,
            normalization_exponent=self.normalization_exponent,
        )

        self.graph.train()
        self.graph = graph
        self.scope = scope

    def step(self, data_train, data_val):
        """
        Run one optimizer step with the batch of training and test data.

        Args:
            data_train (tuple(Tensor, Tensor)): A tuple of input and target
                tensors from the training split
            data_val (tuple(Tensor, Tensor)): A tuple of input and target
                tensors from the validation split
            error_dict

        Returns:
            dict: A dict containing training statistics
        """
        input_train, target_train = data_train
        input_val, target_val = data_val

        input_train = input_train.to(self.device)
        target_train = target_train.to(self.device)
        input_val = input_val.to(self.device)
        target_val = target_val.to(self.device)

        self.graph.train()
        self.op_optimizer.zero_grad()
        logits_train = self.graph(input_train)
        train_loss = self.loss(logits_train, target_train)
        train_loss.backward()  # retain_graph=True)
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.graph.parameters(), self.grad_clip)
        self.op_optimizer.step()

        with torch.no_grad():
            self.graph.eval()
            logits_val = self.graph(input_val)
            val_loss = self.loss(logits_val, target_val)

        return logits_train, logits_val, train_loss, val_loss

    def get_final_architecture(self):
        """
        Returns the final discretized architecture.

        Returns:
            Graph: The final architecture.
        """
        graph = self.graph.clone().unparse()
        graph.prepare_discretization()
        normalization_exponent = self.normalization_exponent

        # Recalculate ZCP scores at the beginning of each epoch
        # self._calculate_and_set_zcp_scores(self.graph, self.scope)

        def update_l2_weights(edge):
            """
            For operations like SepConv etc that contain suboperations like Conv2d() etc. the square of
            l2 norm of the weights is stored in the corresponding weights shared attribute.
            Suboperations like ReLU are ignored as they have no weights of their own.
            For operations (not suboperations) like Identity() etc. that do not have weights,
            the weights attached to them are used.
            """
            if edge.data.has("alpha"):
                weight = 0.0
                group_dim = torch.zeros(1)
                for i in range(len(edge.data.op.primitives)):
                    # Skip pruned slots
                    if getattr(edge.data.op.primitives[i], "was_pruned_slot", False):
                        logger.info(
                            "Skip L2 weight update for pruned slot edge(%s->%s) prim=%d %s",
                            edge.head,
                            edge.tail,
                            i,
                            type(edge.data.op.primitives[i]).__name__,
                        )
                        continue
                    try:
                        for j in range(len(edge.data.op.primitives[i].op)):
                            try:
                                group_dim += torch.numel(
                                    edge.data.op.primitives[i].op[j].weight
                                )
                                weight += (
                                    torch.norm(
                                        edge.data.op.primitives[i].op[j].weight, 2
                                    )
                                    ** 2
                                ).item()
                            except (AttributeError, TypeError) as e:
                                try:
                                    for k in range(
                                        len(edge.data.op.primitives[i].op[j].op)
                                    ):
                                        group_dim += torch.numel(
                                            edge.data.op.primitives[i]
                                            .op[j]
                                            .op[k]
                                            .weight
                                        )
                                        weight += (
                                            torch.norm(
                                                edge.data.op.primitives[i]
                                                .op[j]
                                                .op[k]
                                                .weight,
                                                2,
                                            )
                                            ** 2
                                        ).item()
                                except AttributeError:
                                    continue
                        edge.data.weights[i] += weight
                        edge.data.dimension[i] += group_dim.item()
                        weight = 0.0
                        group_dim = torch.zeros(1)
                    except AttributeError:
                        # Parameter-less op with attached weight (not placeholders)
                        size = torch.tensor(
                            torch.numel(edge.data.op.primitives[i].weight)
                        )
                        edge.data.weights[i] += (
                            edge.data.op.primitives[i].weight.item()
                        ) ** 2
                        edge.data.dimension[i] += size

        def normalize_weights(edge):
            if edge.data.has("alpha"):
                for i in range(len(edge.data.op.primitives)):
                    edge.data.weights[i] = (
                        torch.sqrt(edge.data.weights[i])
                        / torch.pow(
                            edge.data.dimension[i], normalization_exponent
                        ).item()
                    )

        def prune_weights(edge):
            """
            Operations whose l2 norm of the weights across all cells of the same
            type (normal or reduced) is less than the threshold are pruned away.
            To achieve this, the alpha flag for the corresponding operation is
            turned off (replaced with zero).
            """
            if edge.data.has("alpha"):
                for i in range(len(edge.data.weights)):
                    if torch.sqrt(edge.data.weights[i]) < self.threshold:
                        edge.data.alpha[i] = 0

        def reinitialize_l2_weights(edge):
            if edge.data.has("alpha"):
                for i in range(len(edge.data.weights)):
                    edge.data.weights[i] = 0
                    edge.data.dimension[i] = 0

        def discretize_ops(edge):
            if edge.data.has("alpha"):
                primitives = edge.data.op.get_embedded_ops()
                weights = edge.data.weights.detach().cpu()

                # collect candidates: skip placeholders; prefer non-Zero if any

                all_valid_candidates = []
                for idx, prim in enumerate(primitives):
                    if getattr(prim, "was_pruned_slot", False):
                        continue
                    all_valid_candidates.append((idx, float(weights[idx])))

                # fallback to first non-placeholder if somehow empty
                if not all_valid_candidates:
                    for idx, prim in enumerate(primitives):
                        if not getattr(prim, "was_pruned_slot", False):
                            chosen = idx
                            break
                    else:
                        chosen = 0  # very unlikely
                else:
                    chosen = max(all_valid_candidates, key=lambda t: t[1])[0]

                edge.data.set("op", primitives[chosen])

        # Detailed description of the operations are provided in the functions.
        graph.update_edges(update_l2_weights, scope=self.scope, private_edge_data=True)
        graph.update_edges(normalize_weights, scope=self.scope, private_edge_data=True)
        # graph.update_edges(prune_weights, scope=self.scope, private_edge_data=True)

        graph.update_edges(discretize_ops, scope=self.scope, private_edge_data=True)
        graph.update_edges(
            reinitialize_l2_weights, scope=self.scope, private_edge_data=False
        )
        graph.prepare_evaluation()
        graph.parse()
        # graph.QUERYABLE=False
        graph = graph.to(self.device)
        return graph

    def test_statistics(self):
        """
        Return anytime test statistics if provided by the optimizer.
        On the last epoch, this will compute and store the final architecture.
        """
        # nb301 is not there but we use it anyways to generate the arch strings.
        # if self.graph.QUERYABLE:
        try:
            # record anytime performance
            self.best_arch = self.get_final_architecture()
            return (
                self.best_arch.query(
                    Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
                ),
                self.best_arch.query(
                    Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
                ),
                self.best_arch.query(
                    Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
                ),
                self.best_arch.query(
                    Metric.TRAIN_TIME, self.dataset, dataset_api=self.dataset_api
                ),
            )
        except Exception as e:
            logger.error(f"Failed to query anytime performance: {e}")
            return None

    def before_training(self):
        """
        Function called right before training starts. To be used as hook
        for the optimizer.
        """
        """
        Move the graph into cuda memory if available.
        """
        self.graph = self.graph.to(self.device)
        self.operation_weights = self.operation_weights.to(self.device)

        # ZCP_GSparseOptimizer.check_pruning_invariants(
        #     self.op_optimizer, self.graph, self.scope
        # )

    def new_epoch(self, epoch):
        """
        Just log the l2 norms of operation weights.
        """
        if epoch > 0 and self.enable_zcp_scaling:
            self._refresh_group_weight_decay_from_zcp(
                optimizer=self.op_optimizer,
                graph=self.graph,
                scope=self.scope,
                base_mu=self.mu,
            )
        # else:
        #     ZCP_GSparseOptimizer.dump_edge_prims(self.graph, self.scope)

        normalization_exponent = self.normalization_exponent

        def update_l2_weights(edge):
            """
            For operations like SepConv etc that contain suboperations like Conv2d() etc. the square of
            l2 norm of the weights is stored in the corresponding weights shared attribute.
            Suboperations like ReLU are ignored as they have no weights of their own.
            For operations (not suboperations) like Identity() etc. that do not have weights,
            the weights attached to them are used.
            """
            if edge.data.has("alpha"):
                weight = 0.0
                group_dim = torch.zeros(1)
                for i in range(len(edge.data.op.primitives)):
                    # Skip pruned slots
                    if getattr(edge.data.op.primitives[i], "was_pruned_slot", False):
                        continue
                    try:
                        for j in range(len(edge.data.op.primitives[i].op)):
                            try:
                                group_dim += torch.numel(
                                    edge.data.op.primitives[i].op[j].weight
                                )
                                weight += (
                                    torch.norm(
                                        edge.data.op.primitives[i].op[j].weight, 2
                                    )
                                    ** 2
                                ).item()
                            except (AttributeError, TypeError) as e:
                                try:
                                    for k in range(
                                        len(edge.data.op.primitives[i].op[j].op)
                                    ):
                                        group_dim += torch.numel(
                                            edge.data.op.primitives[i]
                                            .op[j]
                                            .op[k]
                                            .weight
                                        )
                                        weight += (
                                            torch.norm(
                                                edge.data.op.primitives[i]
                                                .op[j]
                                                .op[k]
                                                .weight,
                                                2,
                                            )
                                            ** 2
                                        ).item()
                                except AttributeError:
                                    continue
                        edge.data.weights[i] += weight
                        edge.data.dimension[i] += group_dim.item()
                        weight = 0.0
                        group_dim = torch.zeros(1)
                    except AttributeError:
                        # Parameter-less op with attached weight (not placeholders)
                        size = torch.tensor(
                            torch.numel(edge.data.op.primitives[i].weight)
                        )
                        edge.data.weights[i] += (
                            edge.data.op.primitives[i].weight.item()
                        ) ** 2
                        edge.data.dimension[i] += size

        def normalize_weights(edge):
            if edge.data.has("alpha"):
                for i in range(len(edge.data.op.primitives)):
                    edge.data.weights[i] = (
                        torch.sqrt(edge.data.weights[i])
                        / torch.pow(
                            edge.data.dimension[i], normalization_exponent
                        ).item()
                    )

        def reinitialize_l2_weights(edge):
            if edge.data.has("alpha"):
                for i in range(len(edge.data.weights)):
                    edge.data.weights[i] = 0
                    edge.data.dimension[i] = 0

        self.graph.update_edges(
            update_l2_weights, scope=self.scope, private_edge_data=True
        )
        self.graph.update_edges(
            normalize_weights, scope=self.scope, private_edge_data=True
        )

        for alpha in self.graph.get_all_edge_data("weights"):
            self.operation_weights.append(alpha)
        weights_str = [
            ", ".join(["{:+.06f}".format(torch.sqrt(x)) for x in a])
            + ", {}".format(np.max(torch.sqrt(a).detach().cpu().numpy()))
            for a in self.operation_weights
        ]
        logger.info(
            "Arch weights (normalized weights, last column max): \n{}".format(
                "\n".join(weights_str)
            )
        )
        self.graph.update_edges(
            reinitialize_l2_weights, scope=self.scope, private_edge_data=False
        )
        self.operation_weights = torch.nn.ParameterList()
        self.graph.to(self.device)
        super().new_epoch(epoch)

    def after_training(self):
        print("save path: ", self.config.save)
        if not self.best_arch:
            logger.info(
                "Best arch not computed during last epoch's test_statistics. Computing now."
            )
            self.best_arch = self.get_final_architecture()

        logger.info("Final architecture after search:\n" + self.best_arch.modules_str())

    def get_op_optimizer(self):
        """
        This is required for the final validation when
        training from scratch.

        Returns:
            (torch.optim.Optimizer): The optimizer used for the op weights update.
        """
        return self.op_optimizer_evaluate.__class__

    def get_model_size(self):
        return count_parameters_in_MB(self.graph)

    def get_checkpointables(self):
        """
        Return all objects that should be saved in a checkpoint during training.

        Will be called after `before_training` and must include key "model".

        Returns:
            (dict): with name as key and object as value. e.g. graph, arch weights, optimizers, ...
        """
        logger.info(
            "Saving checkpoint with %d pruned architectures.",
            len(self.pruned_op_indices),
        )
        return {
            "model": self.graph,
            "op_optimizer": self.op_optimizer,
            "pruned_op_indices": SimpleStateDict(
                {"pruned_op_indices": self.pruned_op_indices}
            ),
        }


class GSparseMixedOp(MixedOp):
    def __init__(self, primitives, min_cuda_memory=False):
        """
        Initialize the mixed op for Group Sparsity.

        Args:
            primitives (list): The primitive operations to sample from.
        """
        super().__init__(primitives)
        self.min_cuda_memory = min_cuda_memory

    def forward(self, x, edge_data):
        """
        Output of operations like Identity() that do not have weighted suboperations
        like Conv2d(), are multipled with the weight parameter attached to them, so
        that these weights are optimized as well, during the training phase.
        """
        summed = 0
        n_active = 0
        for op in self.primitives:
            if getattr(op, "was_pruned_slot", False):
                continue
            out = op(x, None)
            if hasattr(op, "weight"):
                out = op.weight * out
            summed = out if summed is None else (summed + out)
            n_active += 1
            # Optional: stabilize scale by averaging
            # if n_active > 0:
            #     summed = summed / n_active
        return summed

        # summed = 0
        # for op in self.primitives:
        #     # Skip pruned placeholder slots entirely
        #     if getattr(op, "was_pruned_slot", False):
        #         continue
        #     try:
        #         len(op.op)
        #         summed += op(x, None)
        #     except AttributeError:
        #         if op.training and edge_data.has("alpha"):
        #             summed += op.weight * op(x, None)
        #         else:
        #             summed += op(x, None)

        # # summed = torch.nn.functional.normalize(summed)
        # return summed

    # The following functions are obsolete because of the forward implementation but are needed due to the inheritance
    def get_weights(self, edge_data):
        """
        Return the weights of the operations.
        """
        # return edge_data.alpha  # use pudb to check if this has alpha
        pass

    def process_weights(self, weights):
        """
        Process the weights of the operations.
        """
        # return torch.softmax(
        #     weights, dim=-1
        # )  # or normalize torch.nn.functional.normalize(weights)
        pass

    def apply_weights(self, x, weights):
        """
        Apply the weights to the operations.
        """
        # weighted_sum = sum(w * op(x, None) for w, op in zip(weights, self.primitives))
        # return weighted_sum
        pass
