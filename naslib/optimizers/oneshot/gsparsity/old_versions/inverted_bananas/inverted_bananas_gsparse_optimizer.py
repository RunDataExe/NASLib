#
#!!!! check if resuming from first stage also saves archs that should be pruned
import logging
import torch
import numpy as np
from copy import deepcopy  # Ensure deepcopy is imported
from pathlib import Path
import json
import os  # Import os for path checking

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
from naslib.utils import SimpleStateDict  # Import the new wrapper

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
            "Not sure if the resumption of the list of architectures that should be removed works, when resuming from the first stage. If this is needed verify first."
        )
        logger.info(
            "Not sure if the removal of the archs is shifting the indices of the architectures in the search space, if this is needed verify first."
        )
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
        self.ibgs_state_wrapper = SimpleStateDict()  # Instantiate the wrapper

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
        return False  # Default if stage is not set

    def before_training(self, resume_from_path=None):
        logger.info("Calling before_training for Inverted_Bananas_GsparseOptimizer.")

        # Default to Stage 1, will be overridden if resuming
        self.current_stage = 1
        # self.worst_architectures_op_indices is initialized as []
        # self.ibgs_state_wrapper.state is initially {}
        logger.info("Trying to resume from path: {}".format(resume_from_path))
        if resume_from_path and os.path.exists(resume_from_path):
            logger.info(f"Attempting to resume from checkpoint: {resume_from_path}")
            checkpoint_data = torch.load(resume_from_path, map_location="cpu")

            # Load Inverted_Bananas_GsparseOptimizer specific state if present in the checkpoint
            # The key in checkpoint_data will be "ibgs_specific_state_wrapper" (from get_checkpointables)
            # and its value will be the dictionary returned by SimpleStateDict.state_dict()
            loaded_from_wrapper = False
            if "ibgs_specific_state_wrapper" in checkpoint_data:
                loaded_ibgs_state_dict = checkpoint_data["ibgs_specific_state_wrapper"]
                if (
                    isinstance(loaded_ibgs_state_dict, dict)
                    and loaded_ibgs_state_dict.get("optimizer_name")
                    == "Inverted_Bananas_GsparseOptimizer"
                ):
                    self.current_stage = loaded_ibgs_state_dict.get(
                        "current_stage_val", 1
                    )
                    self.worst_architectures_op_indices = loaded_ibgs_state_dict.get(
                        "worst_architectures_op_indices_val", []
                    )
                    # Also update the wrapper's state, which will be used by the checkpointer
                    self.ibgs_state_wrapper.load_state_dict(loaded_ibgs_state_dict)
                    logger.info(
                        f"Loaded IBGS specific state from wrapper in checkpoint: current_stage={self.current_stage}, "
                        f"{len(self.worst_architectures_op_indices)} worst archs identified."
                    )
                    loaded_from_wrapper = True
                else:
                    logger.warning(
                        "Found 'ibgs_specific_state_wrapper' in checkpoint, but content mismatch or not a dict. Optimizer name: {}".format(
                            loaded_ibgs_state_dict.get("optimizer_name")
                            if isinstance(loaded_ibgs_state_dict, dict)
                            else "N/A"
                        )
                    )

            if not loaded_from_wrapper:
                # Fallback: Infer stage from checkpoint content if IBGS state is not found or invalid
                logger.info(
                    "IBGS specific state not found/valid in checkpoint via wrapper. Inferring stage from content."
                )
                self._infer_stage_from_checkpoint_content(checkpoint_data)

            if self.current_stage == 2:
                logger.info("Resuming into Stage 2.")
                if self.worst_architectures_op_indices:
                    logger.info(
                        f"Re-applying pruning of {len(self.worst_architectures_op_indices)} architectures to self.search_space."
                    )
                    for arch_op_idx_raw in self.worst_architectures_op_indices:
                        try:
                            arch_op_idx_cleaned = [int(val) for val in arch_op_idx_raw]
                            remove_architecture(
                                self.search_space,
                                arch_representation=arch_op_idx_cleaned,
                                representation_type="op_indices",
                                scope=self.scope,
                            )
                        except Exception as e_prune:
                            logger.warning(
                                f"Could not re-apply pruning for architecture {arch_op_idx_raw} during resume: {e_prune}"
                            )
                    logger.info(
                        "Pruning re-applied to self.search_space for Stage 2 resume."
                    )
                else:
                    logger.warning(
                        "Resuming into Stage 2, but no 'worst_architectures_op_indices' were loaded/found. "
                        "GSParseOptimizer will use the search space as is."
                    )

                logger.info(
                    "Adapting search space for Stage 2 optimizer (GSParseOptimizer) during resume."
                )
                self.stage2_optimizer.adapt_search_space(self.search_space, self.scope)
                self.stage2_optimizer.before_training()  # Call before_training for stage2_optimizer

            elif self.current_stage == 1:
                logger.info("Resuming into Stage 1.")
                self.stage1_optimizer.before_training()  # Call before_training for stage1_optimizer

        else:  # No resume path or path does not exist
            if resume_from_path:
                logger.warning(
                    f"Resume path {resume_from_path} not found. Starting fresh as Stage 1."
                )
            else:
                logger.info("No resume path provided. Starting fresh as Stage 1.")
            self.current_stage = 1
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
            # Ensure self.search_space is the one with betas
            self.stage2_optimizer.adapt_search_space(
                self.search_space, self.scope
            )  # Pass the (potentially unpruned if history empty) search_space
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
                    self.search_space,  # This modifies self.search_space in place
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
        # Now self.search_space is pruned. Pass this to stage2_optimizer.
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
            # Return the actual optimizer instance used by the GSparseOptimizer (stage 2)
            # The GSparseOptimizer stores its search optimizer in self.op_optimizer
            if (
                hasattr(self.stage2_optimizer, "op_optimizer")
                and self.stage2_optimizer.op_optimizer is not None
            ):
                return self.stage2_optimizer.op_optimizer
            else:
                logger.warning(
                    "Stage 2 optimizer (GSParseOptimizer) does not have a configured 'op_optimizer' instance."
                )
                return None
        # For Stage 1 (Inverted_Bananas), or if Stage 2 optimizer is not yet fully set up,
        # this optimizer does not provide an op_optimizer for the trainer to manage directly
        # in the same way as step-based optimizers.
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
        current_stage_checkpointables = {}
        if self.current_stage == 1:
            if hasattr(self.stage1_optimizer, "get_checkpointables"):
                current_stage_checkpointables = (
                    self.stage1_optimizer.get_checkpointables()
                )
            else:
                logger.warning(
                    "Stage 1 optimizer does not implement get_checkpointables."
                )
        elif self.current_stage == 2:
            if hasattr(self.stage2_optimizer, "get_checkpointables"):
                current_stage_checkpointables = self.stage2_optimizer.get_checkpointables()  # This should include {'model': gsparse_graph, 'op_optimizer': gsparse_op_optimizer}
            else:
                logger.warning(
                    "Stage 2 optimizer does not implement get_checkpointables."
                )
        else:
            logger.warning(
                f"get_checkpointables called with unknown stage: {self.current_stage}"
            )

        # Update the state of the wrapper instance before returning it
        # This ensures the wrapper itself is saved with the latest IBGS state.
        self.ibgs_state_wrapper.state = {
            "optimizer_name": "Inverted_Bananas_GsparseOptimizer",  # For identification
            "current_stage_val": self.current_stage,
            "worst_architectures_op_indices_val": self.worst_architectures_op_indices,
        }

        final_checkpointables = {**current_stage_checkpointables}
        # Add the wrapper instance itself as a checkpointable object
        # Its state_dict will be called by the fvcore.Checkpointer
        final_checkpointables["ibgs_specific_state_wrapper"] = self.ibgs_state_wrapper
        return final_checkpointables

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
