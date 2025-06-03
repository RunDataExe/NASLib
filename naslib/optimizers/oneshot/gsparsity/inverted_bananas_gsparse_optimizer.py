import logging
import torch
import numpy as np
from copy import deepcopy
from pathlib import Path
import json

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.oneshot.gsparsity.gsparsity_optimizer import GSparseOptimizer
from naslib.optimizers.oneshot.gsparsity.inverted_bananas_optimizer import (
    Inverted_Bananas,
)
from naslib.optimizers.oneshot.gsparsity.remove_arch_from_search_space import (
    add_betas_to_edges,
    remove_architecture,
)
from naslib.utils import get_zc_benchmark_api
from naslib.search_spaces.core.query_metrics import Metric
# Assuming NasBench201SearchSpace or similar that has get_op_indices()
# from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_op_indices # Not needed if arch.get_op_indices() works

logger = logging.getLogger(__name__)


class Inverted_Bananas_GsparseOptimizer(MetaOptimizer):
    """
    A two-stage optimizer that first removes poor architectures using Inverted BANANAS,
    then applies GSparsity to the reduced search space.

    Stage 1: Identify and remove worst architectures using Inverted BANANAS.
    Stage 2: Apply GSparseOptimizer on the pruned search space.
    """

    def __init__(self, config):
        super(Inverted_Bananas_GsparseOptimizer, self).__init__()
        self.config = config
        self.dataset = config.dataset

        # Stage 1: Inverted BANANAS configuration
        self.stage1_config = deepcopy(config.stage1)
        self.stage1_config.dataset = config.dataset
        self.stage1_epochs = self.stage1_config.search.epochs
        self.removal_percentage = self.stage1_config.search.removal_percentage

        logger.info(
            f"Stage 1 (Inverted BANANAS) will run for {self.stage1_epochs} epochs."
        )
        logger.info(
            f"Will remove {self.removal_percentage * 100:.2f}% of architectures after Stage 1."
        )
        self.stage1_optimizer = Inverted_Bananas(self.stage1_config)

        # Stage 2: GSParseOptimizer configuration
        self.stage2_config = deepcopy(config.stage2)
        self.stage2_config.dataset = config.dataset
        self.stage2_epochs = self.stage2_config.search.epochs
        logger.info(
            f"Stage 2 (GSParseOptimizer) will run for {self.stage2_epochs} epochs."
        )
        self.stage2_optimizer = GSparseOptimizer(self.stage2_config)

        self.current_stage = 1
        self.current_overall_epoch = 0

        self.search_space = None
        self.scope = None
        self.dataset_api = None

        self.worst_architectures_op_indices = []

        # self.using_step_function = False

    def adapt_search_space(
        self, search_space, scope=None, dataset_api=None, train_loader=None
    ):
        # super().adapt_search_space(search_space, scope, dataset_api)
        self.search_space = search_space
        self.scope = scope
        self.dataset_api = dataset_api

        logger.info("Adding beta parameters to edges for potential pruning.")
        add_betas_to_edges(self.search_space, scope=self.scope)

        # Adapt search space for stage 1 optimizer
        # Give a deepcopy to stage1, as BANANAS might have internal state tied to it,
        # though it's query-based and shouldn't modify the graph structure.
        # The main self.search_space is the one that will be pruned.
        self.stage1_optimizer.adapt_search_space(
            deepcopy(self.search_space), scope=self.scope, dataset_api=self.dataset_api
        )
        # Stage 2 optimizer will be adapted later with the pruned search space.

    @property
    def using_step_function(self):
        if self.current_stage == 1:
            return self.stage1_optimizer.using_step_function
        elif self.current_stage == 2:
            return self.stage2_optimizer.using_step_function

    def before_training(self):
        logger.info("Calling before_training for Inverted_Bananas_GsparseOptimizer.")
        self.stage1_optimizer.before_training()

    def new_epoch(self, epoch):
        self.current_overall_epoch = epoch
        logger.debug(f"Overall Epoch: {epoch}. Current Stage: {self.current_stage}")

        if self.current_stage == 1 and epoch == self.stage1_epochs:
            logger.info(
                f"Overall epoch {epoch} reached. This is the designated start for Stage 2."
            )
            self._perform_transition_to_stage2()

        if self.current_stage == 1:
            if epoch < self.stage1_epochs:
                self.stage1_optimizer.new_epoch(epoch)
            # Transition logic is handled in train_statistics after the last epoch of stage 1

        elif self.current_stage == 2:
            stage2_epoch = epoch - self.stage1_epochs
            self.stage2_optimizer.new_epoch(stage2_epoch)

    def _perform_transition_to_stage2(self):
        logger.info(
            f"Transitioning from Stage 1 to Stage 2. Stage 1 completed {self.stage1_epochs} epochs."
        )

        # Convert ModuleList to a regular list to use sort()
        evaluated_architectures_meta = list(self.stage1_optimizer.history)
        if not evaluated_architectures_meta:
            logger.warning(
                "No architectures found in Inverted BANANAS history. Skipping pruning."
            )
            self.current_stage = 2
            logger.info(
                "Adapting search space and calling before_training for Stage 2 (GSParseOptimizer) with unpruned space."
            )
            self.stage2_optimizer.adapt_search_space(
                self.search_space, self.scope, self.dataset_api
            )
            self.stage2_optimizer.before_training()
            return

        evaluated_architectures_meta.sort(
            key=lambda x: x.accuracy
        )  # Ascending for Inverted BANANAS

        num_to_remove = int(self.removal_percentage * len(evaluated_architectures_meta))
        worst_architectures_meta = evaluated_architectures_meta[:num_to_remove]

        logger.info(
            f"Identified {len(worst_architectures_meta)} worst architectures to remove."
        )

        self.worst_architectures_op_indices = []
        for arch_meta in worst_architectures_meta:
            try:
                op_indices = arch_meta.arch.get_op_indices()
                self.worst_architectures_op_indices.append(op_indices)
                logger.debug(
                    f"Marking for removal: {op_indices} (Accuracy: {arch_meta.accuracy})"
                )
            except Exception as e:
                logger.error(
                    f"Could not get op_indices for an architecture: {e}. Arch: {arch_meta.arch}"
                )

        logger.info(
            f"Starting pruning of {len(self.worst_architectures_op_indices)} architectures from the main search space."
        )
        for i, arch_op_idx in enumerate(self.worst_architectures_op_indices):
            logger.debug(
                f"Removing architecture {i + 1}/{len(self.worst_architectures_op_indices)}: {arch_op_idx}"
            )
            try:
                remove_architecture(
                    self.search_space,
                    arch_op_idx,
                    representation_type="op_indices",
                    scope=self.scope,
                )
            except IndexError as e:
                logger.warning(
                    f"Could not remove architecture {arch_op_idx} (already removed or op not found?): {e}"
                )
            except Exception as e:
                logger.error(f"Error removing architecture {arch_op_idx}: {e}")
        logger.info("Pruning complete.")

        self.current_stage = 2
        logger.info(
            "Adapting search space and calling before_training for Stage 2 (GSParseOptimizer)."
        )
        self.stage2_optimizer.adapt_search_space(self.search_space, self.scope)
        self.stage2_optimizer.before_training()

    def train_statistics(self, report_incumbent=True):
        if self.current_stage == 1:
            return self.stage1_optimizer.train_statistics(report_incumbent)
        else:
            logger.error(f"Invalid stage: {self.current_stage}")
            # Return dummy/empty statistics to avoid crashing trainer
            logger.warning(
                "Returning dummy statistics for invalid stage in train_statistics."
            )
            return 0.0, 0.0, 0.0, 0.0  # train_acc, valid_acc, test_acc, train_time

    def step(self, data_train, data_val):
        """
        Delegates the step call to the current stage's optimizer.
        This method is expected to be called only when self.using_step_function is True,
        which corresponds to Stage 2 (GSParseOptimizer).
        """
        if self.current_stage == 2:
            return self.stage2_optimizer.step(data_train, data_val)
        else:
            # This case should ideally not be reached if the Trainer respects using_step_function.
            # If stage 1 (Inverted_Bananas) were to use step, it would be handled here.
            # However, Inverted_Bananas (via Bananas) sets using_step_function = False.
            logger.error(
                f"Step function called unexpectedly for stage {self.current_stage}."
            )
            # Call super().step() which will raise NotImplementedError from MetaOptimizer,
            # indicating an issue with the control flow if this path is taken.
            return super().step(data_train, data_val)

    def test_statistics(self):
        if self.current_stage == 1:
            return self.stage1_optimizer.test_statistics()
        elif self.current_stage == 2:
            return self.stage2_optimizer.test_statistics()
        else:
            logger.error(f"Invalid stage: {self.current_stage} in test_statistics")
            return None  # Or appropriate default

    def get_final_architecture(self):
        if self.current_stage == 2:
            final_arch_stage2 = self.stage2_optimizer.get_final_architecture()
            if final_arch_stage2 is not None:
                logger.info(
                    "Getting final architecture from Stage 2 (GSParseOptimizer)."
                )
                return final_arch_stage2
            else:
                logger.warning(
                    "Stage 2 (GSParseOptimizer) did not yield a final architecture."
                )

        logger.warning(
            "Falling back to Stage 1 (Inverted BANANAS) for final architecture (inverted perspective)."
        )
        if self.stage1_optimizer.history:
            return (
                self.stage1_optimizer.get_final_architecture()
            )  # Returns the worst one found

        logger.error("No final architecture available from either stage.")
        return None

    def get_op_optimizer(self):
        if self.current_stage == 2:
            if hasattr(self.stage2_optimizer, "op_optimizer"):
                return self.stage2_optimizer.op_optimizer
            elif hasattr(
                self.stage2_optimizer, "get_op_optimizer"
            ):  # Some optimizers might have a getter
                return self.stage2_optimizer.get_op_optimizer()
        return None

    def get_model_size(self):
        # The model size is primarily determined by the search space structure.
        # If sub-optimizers report different sizes based on internal parameters,
        # this could be stage-dependent. However, for NASLib, it's usually search_space.n_params()
        if self.search_space and hasattr(self.search_space, "get_model_size"):
            return (
                self.search_space.get_model_size()
            )  # Prefer search_space's own method if available
        if self.current_stage == 1 and hasattr(self.stage1_optimizer, "get_model_size"):
            return self.stage1_optimizer.get_model_size()
        elif self.current_stage == 2 and hasattr(
            self.stage2_optimizer, "get_model_size"
        ):
            return self.stage2_optimizer.get_model_size()

    def get_checkpointables(self):
        # checkpointables = super().get_checkpointables()
        return {
            "current_stage": self.current_stage,
            "model": self.stage2_optimizer.get_checkpointables()["model"]
            if self.current_stage == 2
            else self.stage1_optimizer.get_checkpointables()["model"],
            "current_overall_epoch": self.current_overall_epoch,
            "worst_architectures_op_indices": self.worst_architectures_op_indices,
            "stage1_optimizer_state": self.stage1_optimizer.get_checkpointables()
            if hasattr(self.stage1_optimizer, "get_checkpointables")
            else {},
            "stage2_optimizer_state": self.stage2_optimizer.get_checkpointables()
            if hasattr(self.stage2_optimizer, "get_checkpointables")
            else {},
        }

    # def load_checkpointables(self, checkpointables):
    #     super().load_checkpointables(checkpointables)
    #     self.current_stage = checkpointables.get("current_stage", 1)
    #     self.current_overall_epoch = checkpointables.get("current_overall_epoch", 0)
    #     self.worst_architectures_op_indices = checkpointables.get(
    #         "worst_architectures_op_indices", []
    #     )

    #     logger.info(
    #         f"Loading checkpoint. Resuming at stage {self.current_stage}, overall epoch {self.current_overall_epoch}."
    #     )

    #     stage1_state = checkpointables.get("stage1_optimizer_state")
    #     if hasattr(self.stage1_optimizer, "load_checkpointables") and stage1_state:
    #         self.stage1_optimizer.load_checkpointables(stage1_state)

    #     stage2_state = checkpointables.get("stage2_optimizer_state")
    #     if hasattr(self.stage2_optimizer, "load_checkpointables") and stage2_state:
    #         self.stage2_optimizer.load_checkpointables(stage2_state)

    #     # Important: The search_space itself is checkpointed by the Trainer.
    #     # After it's loaded by the Trainer, we need to ensure it's correctly set up here.
    #     if self.search_space:
    #         add_betas_to_edges(self.search_space, scope=self.scope)  # Idempotent

    #         # If resuming into stage 2, or after pruning was supposed to happen,
    #         # ensure stage2_optimizer is adapted with the current self.search_space
    #         # (which should be the pruned one if loaded correctly by trainer).
    #         if self.current_stage == 2:
    #             logger.info(
    #                 "Resuming in Stage 2. Re-adapting Stage 2 optimizer with loaded search space."
    #             )
    #             self.stage2_optimizer.adapt_search_space(
    #                 self.search_space,
    #                 self.scope,
    #                 self.dataset_api,
    #             )
    #             # Calling before_training() again might be necessary if its state wasn't fully captured
    #             # or if it needs to re-initialize based on the potentially modified search space.
    #             # However, this could also reset parts of its loaded state.
    #             # This depends heavily on GSParseOptimizer's checkpointing and before_training logic.
    #             # For now, we assume load_checkpointables + adapt_search_space is sufficient.
    #             # self.stage2_optimizer.before_training() # Use with caution
    #         elif (
    #             self.current_stage == 1
    #             and self.current_overall_epoch >= self.stage1_epochs
    #         ):
    #             # This case means we loaded a checkpoint that was saved *after* stage 1 finished
    #             # but *before* stage 2 formally started its first epoch via new_epoch.
    #             # The pruning should have occurred. We should ensure we are in stage 2.
    #             logger.info(
    #                 f"Resuming after Stage 1 ({self.current_overall_epoch}/{self.stage1_epochs - 1} epochs done). Ensuring transition to Stage 2."
    #             )
    #             if (
    #                 not self.worst_architectures_op_indices
    #                 and self.current_overall_epoch == self.stage1_epochs - 1
    #             ):
    #                 # If worst_architectures_op_indices is empty, it implies pruning might not have been saved/done yet.
    #                 # This is a tricky state. For simplicity, if we are at the boundary, let train_statistics trigger it.
    #                 pass
    #             else:  # Pruning info is available or we are past the point.
    #                 self.current_stage = 2
    #                 self.stage2_optimizer.adapt_search_space(
    #                     self.search_space, self.scope, self.dataset_api
    #                 )
    #                 # self.stage2_optimizer.before_training() # Caution
