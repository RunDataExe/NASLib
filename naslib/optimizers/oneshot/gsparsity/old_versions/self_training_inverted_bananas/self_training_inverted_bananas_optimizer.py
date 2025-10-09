import logging
import numpy as np
import torch

from naslib.optimizers.oneshot.gsparsity.old_versions.self_training_inverted_bananas.self_training_bananas_optimizer import (
    Bananas,
)

logger = logging.getLogger(__name__)


class Inverted_Bananas(Bananas):
    """
    Bananas optimizer that searches for the worst architectures in the search space.
    Inherits from the original Bananas optimizer and overrides functions to reverse
    the optimization direction.
    """

    def __init__(self, config, zc_api=None):
        super().__init__(config, zc_api)
        logger.info(
            "Using Inverted Bananas optimizer that searches for the worst architectures"
        )

    def _get_best_candidates(self, candidates, acq_fn):
        """Override to select candidates with the lowest acquisition function values"""
        if self.zc and len(self.train_data) <= self.max_zerocost:
            for model in candidates:
                model.zc_scores = self.query_zc_scores(model.arch)

            values = [
                acq_fn(model.arch, [{"zero_cost_scores": model.zc_scores}])
                for model in candidates
            ]
        else:
            values = [acq_fn(model.arch) for model in candidates]

        sorted_indices = np.argsort(values)
        # Take the k WORST candidates (lowest acquisition function values)
        choices = [candidates[i] for i in sorted_indices[: self.k]]

        return choices

    def _update_history(self, child):
        """Override to keep the worst architectures in history"""
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy < p.accuracy:
                    self.history[i] = child
                    break

    def get_final_architecture(self):
        """Override to return the architecture with the lowest accuracy"""
        return min(self.history, key=lambda x: x.accuracy).arch
