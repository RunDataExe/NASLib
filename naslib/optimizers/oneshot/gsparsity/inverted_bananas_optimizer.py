import logging
import numpy as np
import torch

from naslib.optimizers.discrete.bananas.optimizer import Bananas

logger = logging.getLogger(__name__)

class Inverted_Bananas(Bananas):
    """
    Bananas optimizer that searches for the worst architectures in the search space.
    Inherits from the original Bananas optimizer and overrides functions to reverse
    the optimization direction.
    """

    # Enable step function compatibility
    # using_step_function = True
    
    def __init__(self, config, zc_api=None):
        super().__init__(config, zc_api)
        logger.info("Using Inverted Bananas optimizer that searches for the worst architectures")
        
        # # Add a dummy optimizer to satisfy the trainer's requirements
        # # This won't actually be used for training since we override the step method
        # import torch
        # self.op_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        
    # def step(self, data_train, data_val):
    #     """
    #     A minimal step function implementation that makes Inverted_Bananas compatible with 
    #     the step-based training approach. This doesn't actually use the dataloaders
    #     but allows the optimizer to work with the Trainer's step-based workflow.
        
    #     Args:
    #         data_train: Ignored in this optimizer
    #         data_val: Ignored in this optimizer
            
    #     Returns:
    #         Dummy tensors expected by the trainer
    #     """
    #     # Create dummy tensors to satisfy the trainer's expectations
    #     batch_size = data_train[0].size(0) if isinstance(data_train, tuple) else 1
    #     classes = 10  # Default, most datasets have at least 10 classes
        
    #     # Create dummy logits and loss values
    #     dummy_logits = torch.zeros(batch_size, classes).to(
    #         data_train[0].device if isinstance(data_train, tuple) else torch.device('cpu')
    #     )
    #     dummy_loss = torch.tensor(0.0, requires_grad=True)
        
    #     return dummy_logits, dummy_logits, dummy_loss, dummy_loss

    def _get_best_candidates(self, candidates, acq_fn):
        """Override to select candidates with the lowest acquisition function values"""
        if self.zc and len(self.train_data) <= self.max_zerocost:
            for model in candidates:
                model.zc_scores = self.query_zc_scores(model.arch)

            values = [acq_fn(model.arch, [{'zero_cost_scores': model.zc_scores}]) for model in candidates]
        else:
            values = [acq_fn(model.arch) for model in candidates]

        sorted_indices = np.argsort(values)
        # Take the k WORST candidates (lowest acquisition function values)
        choices = [candidates[i] for i in sorted_indices[:self.k]]

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