import os
import logging
import sys
import naslib as nl
from fvcore.common.config import CfgNode
import json
import argparse

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    RandomSearch,
    LocalSearch,
    Bananas,
    GSparseOptimizer,
    DrNASOptimizer,
)

from naslib.search_spaces import NasBench201SearchSpace, NasBench301SearchSpace
from naslib import utils
from naslib.utils import setup_logger, get_dataset_api, get_project_root
from naslib.search_spaces.core.query_metrics import Metric

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run NASLib optimizer with specified configuration."
)
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
    help="Optimizer type (e.g., rs, ls, bananas, drnas, gsparsity)",
)
parser.add_argument(
    "--search_space",
    type=str,
    required=True,
    help="Search space type (e.g., nasbench201, nasbench301)",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Dataset (e.g., cifar10, cifar100, imagenet16-120)",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="naslib/optimizers/oneshot/gsparsity/test",
    help="Output directory",
)
parser.add_argument("--epochs", type=int, default=100, help="Number of search epochs")

args = parser.parse_args()

# Use parsed arguments
optimizer_type = args.optimizer
search_space_type = args.search_space
dataset = args.dataset
seed = args.seed
out_dir = args.out_dir
epochs = args.epochs

# TODO how to treat cutout for gsparsity? Does it influence on NASbench301? Also how to treat with respect to hpo? If it does not influence but hpo is wasting time on it would be tragic
# TODO Checkpointing and resuming
# TODO implement the hyperparameters for all (further down is a list)
# TODO HPO using optuna -> setting budget (time limit); optuna.logging and optuna.importance (https://medium.com/@mdshah930/understranding-hyperparameter-importance-in-optuna-mastering-optuna-part-2-c5c88956152a)
# TODO make optuna resumable (syncing NASLib and Optuna)
# TODO make optuna also reproducible
# TODO improve transparend, unbiased seed generation
# TODO add argparse to the script such that it can ealily be automatically run from automatically generated bash scripts using slurm (verification if already runs, or finished)
# TODO add the new implemented methods
# TODO how to handle zcp

# Common evaluation configuration for all optimizers
evaluation = {
    "checkpoint_freq": 30,
    "batch_size": 96,
    "learning_rate": 0.025,
    "learning_rate_min": 0.00,
    "momentum": 0.9,
    "weight_decay": 0.0003,
    "epochs": 600,  #! originally 600
    "warm_start_epochs": 0,
    "grad_clip": 5,
    "train_portion": 1.0,
    "data_size": 50000,
    "cutout": True,
    "cutout_length": 16,
    "cutout_prob": 1.0,
    "drop_path_prob": 0.2,
    "auxiliary_weight": 0.4,
}

# Define optimizer-specific configurations
#! Hyper parameters of each method:
# %! GSparseOptimizer
# % op_optimizer: torch.optim.Optimizer = ProxSGD,
# % op_optimizer_evaluate: torch.optim.Optimizer = torch.optim.SGD,
# % loss_criteria=torch.nn.CrossEntropyLoss(),

# % super(GSparseOptimizer, self).__init__()
# % self.op_optimizer = op_optimizer
# % self.op_optimizer_evaluate = op_optimizer_evaluate
# % self.loss = loss_criteria
# % self.dataset = config.dataset
# % self.grad_clip = config.search.grad_clip
# % self.mu = config.search.weight_decay
# % self.threshold = config.search.threshold
# % self.normalization = config.search.normalization
# % self.normalization_exponent = config.search.normalization_exponent


# %! BANANAS seems like it already allows ZCP?
# % self.config = config
# % self.epochs = config.search.epochs

# % self.performance_metric = Metric.VAL_ACCURACY
# % self.dataset = config.dataset

# % self.k = config.search.k
# % self.num_init = config.search.num_init
# % self.num_ensemble = config.search.num_ensemble
# % self.predictor_type = config.search.predictor_type
# % self.acq_fn_type = config.search.acq_fn_type
# % self.acq_fn_optimization = config.search.acq_fn_optimization
# % self.encoding_type = config.search.encoding_type  # currently not implemented
# % self.num_arches_to_mutate = config.search.num_arches_to_mutate
# % self.max_mutations = config.search.max_mutations
# % self.num_candidates = config.search.num_candidates
# % self.max_zerocost = 1000

# % self.train_data = []
# % self.next_batch = []
# % self.history = torch.nn.ModuleList()

# % self.zc = config.search.zc if hasattr(config.search, 'zc') else None
# % self.semi = "semi" in self.predictor_type
# % self.zc_api = zc_api
# % self.use_zc_api = config.search.use_zc_api if hasattr(
# %     config.search, 'use_zc_api') else False
# % self.zc_names = config.search.zc_names if hasattr(
# %     config.search, 'zc_names') else None
# % self.zc_only = config.search.zc_only if hasattr(
# %     config.search, 'zc_only') else False


# %! LocalSearch
# % self.config = config
# % self.epochs = config.search.epochs

# % self.performance_metric = Metric.VAL_ACCURACY
# % self.dataset = config.dataset

# % self.num_init = config.search.num_init
# % self.nbhd = []
# % self.chosen = None
# % self.best_arch = None

# % self.history = torch.nn.ModuleList()
# % self.newest_child_idx = -1


# %! RandomSearch
# % somewhere I have to be able to set the numb of architectures looked at or does it just increase the num of sampled_archs[] till i stop maybe it is the 100 set here?

# % def _update_history(self, child):
# % if len(self.history) < 100:
# %     self.history.append(child)
# % else:
# %     for i, p in enumerate(self.history):
# %         if child.accuracy > p.accuracy:
# %             self.history[i] = child
# %             break

# % self.performance_metric = Metric.VAL_ACCURACY
# % self.dataset = config.dataset
# % self.fidelity = config.search.fidelity
# % self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# % self.sampled_archs = []
# % self.history = torch.nn.ModuleList()

# %! Random Sampling
# % no Hyperparameters

# %! DRNAS
# % learning_rate: float = 0.025,
# % momentum: float = 0.9,
# % weight_decay: float = 0.0003,
# % grad_clip: int = 5,
# % unrolled: bool = False,
# % arch_learning_rate: float = 0.0003,
# % arch_weight_decay: float = 0.001,
# % epochs: int = 50,
# % op_optimizer: str = "SGD",
# % arch_optimizer: str = "Adam",
# % loss_criteria: str = "CrossEntropyLoss",
# % **kwargs,
optimizer_configs = {
    "rs": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": epochs,
            "fidelity": -1,
        },
    },
    "ls": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": epochs,
            "num_init": 10,
        },
    },
    "bananas": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": epochs,  # ? #! https://github.com/naszilla/bananas/blob/main/nas_algorithms.py epochs = num_init + (total_queries - num_init) / kepochs = 10 + (150 - 10) / 10 = 24 -> to achieve 150 total architecture evaluations
            "k": 10,
            "num_init": 10,
            "num_ensemble": 5,
            "predictor_type": "mlp",
            "acq_fn_type": "its",
            "acq_fn_optimization": "mutation",
            "encoding_type": "path",
            "num_arches_to_mutate": 1,
            "max_mutations": 1,
            "num_candidates": 100,  # no correspondance but resonable value for aquisition function
        },
    },
    "drnas": {
        "search": {  # ? https://github.com/cc-hpc-itwm/GSparsity/blob/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/drnas-c10-original/search-progressive-exp-20210227-085319/log.txt
            "checkpoint_freq": 5,
            "epochs": epochs,
            "batch_size": 48,  # originally 128 in gs logs 48
            "arch_learning_rate": 0.0006,  # originally 0.0003 in gs logs 0.0006
            "cutout": False,  # originally not here in gs logs False
            "cutout_length": 16,  # originally not here in gs logs 16
            "drop_path_prob": 0.3,  # originally not here in gs logs 0.3
            "grad_clip": 5,  # in gs logs 5
            "learning_rate": 0.1,  # originally 0.025 in gs logs 0.1
            "learning_rate_min": 0.0,  # originally 0.0001 in gs logs 0.0
            "momentum": 0.9,  # originally 0.9 in gs logs 0.9
            "train_portion": 0.5,  # originally 0.95 in gs logs 0.5
        },
    },
    "gsparsity": {  # ? https://github.com/cc-hpc-itwm/GSparsity/tree/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/gsparsity-c10/search-for-cell-lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195113 ; https://github.com/cc-hpc-itwm/GSparsity/blob/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/gsparsity-c10/scaling_div_0.5_accuracy_statistics.txt ; https://github.com/cc-hpc-itwm/GSparsity/blob/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/gsparsity-c10/search-for-cell-lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195113/_log_lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195113.txt ; plus paper
        "search": {
            "checkpoint_freq": 5,
            "epochs": epochs,  # in paper 100
            "grad_clip": 0,  # in paper 0
            "weight_decay": 60,  # original 120 in paper 60
            "threshold": 0.000001,
            "normalization": "div",  # in paper div
            "normalization_exponent": 0.5,  # in paper 0.5
            "learning_rate": 0.001,  # original 0.01 in paper 0.001
            "momentum": 0.8,  # in paper 0.8
            "learning_rate_min": 0.0001,  # in paper 0.0001
            "batch_size": 64,  # original 128; in log 64
            "train_portion": 1,  # originally 0.95 in paper 1
            "cutout": False,  # # in paper False (I think NASLIB only needs this for gsparsity on nasbench201)
            "cutout_length": 16,  # in paper 16 (I think NASLIB only needs this for gsparsity on nasbench201)
            # "cutout_prob": 1.0,  # (I think NASLIB only needs this for gsparsity on nasbench201)
        },
        #! zcp-bananas
        #! zcp-gsparsity
        #! bananas-gsparsity
        #! zcp-bananas-gsparsity
        #! bananas-zcp-gsparsity
        #! zcp-bananas-zcp-gsparsity
    },
}

# Add common evaluation to all optimizer configs
for opt in optimizer_configs:
    optimizer_configs[opt]["evaluation"] = evaluation


def update_config(config, optimizer_type, search_space_type, dataset, seed, out_dir):
    """Update the configuration with experiment-specific settings"""
    # Set dataset name

    if optimizer_type == "gsparsity":
        config.save_arch_weights = False

    config.dataset = dataset

    # Dataset path - FIX: Remove the tuple notation (the parentheses and comma)
    config.data = str(get_project_root()) + "/data"  # path to naslib/data directory

    # Set search space
    config.search_space = search_space_type

    # Set up output directory path
    config.save = f"{out_dir}/{optimizer_type}/{search_space_type}/{dataset}/{seed}"

    # Set seed
    config.search.seed = seed

    return config


def get_valid_dataset_and_classes(search_space_type, requested_dataset=None):
    """
    Returns valid dataset and number of classes based on search space type.
    For nasbench301, enforces cifar10.
    For nasbench201, allows selection from supported datasets.
    """
    valid_datasets = {
        "nasbench201": {"cifar10": 10, "cifar100": 100, "imagenet16-120": 120},
        "nasbench301": {"cifar10": 10},
    }

    # Get available datasets for the selected search space
    available_datasets = valid_datasets.get(search_space_type, {})

    if not available_datasets:
        raise ValueError(f"Search space {search_space_type} not supported")

    # For nasbench301, always use cifar10 regardless of what was requested
    if search_space_type == "nasbench301":
        if requested_dataset != "cifar10":
            print(
                f"Warning: NAS-Bench-301 only supports CIFAR-10. Overriding requested dataset '{requested_dataset}'"
            )
        return "cifar10", 10

    # For other search spaces, check if requested dataset is valid
    if requested_dataset in available_datasets:
        return requested_dataset, available_datasets[requested_dataset]
    else:
        raise ValueError(
            f"Dataset '{requested_dataset}' not supported for {search_space_type}. "
            f"Choose from: {list(available_datasets.keys())}"
        )


def run_optimizer(optimizer_type, search_space_type, dataset, config, seed):
    """Run the optimizer with the specified configuration"""
    # Make the results directories
    os.makedirs(config.save + "/search", exist_ok=True)
    os.makedirs(config.save + "/eval", exist_ok=True)

    # Set up the logger
    logger = setup_logger(config.save + "/log.log")
    logger.setLevel(logging.INFO)

    # Log the configuration
    logger.info(f"Configuration is \n{config}")

    # Set up the seed
    utils.set_seed(seed)

    # Create the search space based on the dataset
    dataset, n_classes = get_valid_dataset_and_classes(search_space_type, dataset)

    # Instantiate the search space
    if search_space_type == "nasbench201":
        search_space = NasBench201SearchSpace(n_classes=n_classes)
    elif search_space_type == "nasbench301":
        search_space = NasBench301SearchSpace(n_classes=n_classes)
    else:
        raise ValueError(f"Search space {search_space_type} not supported")

    # Get the benchmark API
    logger.info("Loading Benchmark API")
    dataset_api = get_dataset_api(search_space_type, dataset)

    # Instantiate the optimizer
    if optimizer_type == "rs":
        optimizer = RandomSearch(config)
    elif optimizer_type == "ls":
        optimizer = LocalSearch(config)
    elif optimizer_type == "bananas":
        optimizer = Bananas(config)
    elif optimizer_type == "drnas":
        optimizer = DrNASOptimizer()
    elif optimizer_type == "gsparsity":
        optimizer = GSparseOptimizer(config)
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Adapt the search space for the specific optimizer
    if optimizer_type == "drnas":
        optimizer.adapt_search_space(search_space=search_space, dataset=dataset)
    elif optimizer_type == "gsparsity":
        optimizer.adapt_search_space(search_space=search_space)
    elif optimizer_type in ["rs", "ls", "bananas"]:
        optimizer.adapt_search_space(search_space=search_space, dataset_api=dataset_api)
    else:
        optimizer.adapt_search_space(search_space=search_space)

    # Create trainer and run search
    trainer = Trainer(optimizer, config, lightweight_output=True)
    trainer.search(report_incumbent=False)

    # Get the search trajectory
    search_trajectory = trainer.search_trajectory
    logger.info(f"Train accuracies: {search_trajectory.train_acc}")
    logger.info(f"Validation accuracies: {search_trajectory.valid_acc}")

    # Evaluate the best model found in search
    best_model_val_acc = trainer.evaluate(
        dataset_api=dataset_api, metric=Metric.VAL_ACCURACY
    )
    logger.info(f"Best model validation accuracy: {best_model_val_acc}")

    # Get the best model architecture
    best_model = optimizer.get_final_architecture()

    return search_trajectory, best_model, best_model_val_acc


def main():
    """Main function to run the optimizer"""
    # Create base configuration from the selected optimizer
    if optimizer_type in optimizer_configs:
        config = optimizer_configs[optimizer_type]
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Convert dictionary to CfgNode
    config = CfgNode.load_cfg(json.dumps(config))

    # Automatically select valid dataset and classes
    valid_dataset, n_classes = get_valid_dataset_and_classes(search_space_type, dataset)

    # Process seeds (can be a single seed or a list)
    seeds = [seed] if isinstance(seed, int) else seed

    # Run optimizer for all seeds
    for s in seeds:
        # Update the configuration
        config = update_config(
            config, optimizer_type, search_space_type, valid_dataset, s, out_dir
        )

        # Run the optimizer
        search_trajectory, best_model, best_val_acc = run_optimizer(
            optimizer_type, search_space_type, valid_dataset, config, s
        )

        print(f"Completed run for seed {s}")
        print(f"Best validation accuracy: {best_val_acc}")


if __name__ == "__main__":
    main()
