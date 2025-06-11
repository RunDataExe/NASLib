import logging
import torch
import numpy as np
from copy import deepcopy  # Ensure deepcopy is imported
from pathlib import Path
import json

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.oneshot.gsparsity.gsparsity_optimizer import GSparseOptimizer
from naslib.optimizers.oneshot.gsparsity.inverted_bananas_optimizer import (
    Inverted_Bananas,
)
from naslib.utils.remove_arch_from_search_space import (
    add_betas_to_edges,
    remove_architecture,
)
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
        self.stage1_optimizer.performance_metric = Metric.TRAIN_ACCURACY

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
            self.stage2_optimizer.adapt_search_space(self.search_space, self.scope)
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
        for i, arch_op_idx_raw in enumerate(self.worst_architectures_op_indices):
            logger.debug(
                f"Attempting to remove architecture {i + 1}/{len(self.worst_architectures_op_indices)}: {arch_op_idx_raw}"
            )
            try:
                # Ensure arch_op_idx is a list of Python ints
                # This conversion will handle cases like list of numpy.int64 or list of 0-dim tensors
                arch_op_idx_cleaned = [int(val) for val in arch_op_idx_raw]

                remove_architecture(
                    self.search_space,
                    arch_representation=arch_op_idx_cleaned,  # Use the cleaned version
                    representation_type="op_indices",
                    scope=self.scope,
                )
            except IndexError as e:
                logger.warning(
                    f"Could not remove architecture {arch_op_idx_raw} (already removed or op not found?): {e}"
                )
            except Exception as e:
                logger.error(f"Error removing architecture {arch_op_idx_raw}: {e}")
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
        if self.current_stage == 1:
            return {
                "model": self.stage1_optimizer.get_checkpointables()["model"],
            }
        elif self.current_stage == 2:
            return {
                "model": self.stage2_optimizer.get_checkpointables()["model"],
                "op_optimizer": self.stage2_optimizer.get_checkpointables()[
                    "op_optimizer"
                ],
                "op_optimizer_evaluate": self.stage2_optimizer.get_checkpointables()[
                    "op_optimizer_evaluate"
                ],
            }

    def after_training(self):
        """
        Called after the search process is finished.
        Delegates to the Stage 2 optimizer's after_training method.
        """
        if self.current_stage == 2 and hasattr(self.stage2_optimizer, "after_training"):
            logger.info(
                "Calling after_training for Stage 2 optimizer (GSParseOptimizer)."
            )
            self.stage2_optimizer.config.save = self.config.save
            self.stage2_optimizer.after_training()
        else:
            logger.info(
                "No specific after_training actions for the current state of Inverted_Bananas_GsparseOptimizer or Stage 2 optimizer does not have after_training."
            )

    def get_total_epochs(self):
        """
        Returns the total number of epochs this optimizer will run for.
        """
        return self.stage1_epochs + self.stage2_epochs
