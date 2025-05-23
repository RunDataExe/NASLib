import logging
import os
import sys
import naslib as nl
from fvcore.common.config import CfgNode
import json
import argparse

import optuna
import copy
# from optuna import DEHBSampler, DEHBPruner
import time

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    RandomSearch,
    LocalSearch,
    Bananas,
    # GSparseOptimizer,
    DrNASOptimizer,
    ZCP_GSparseOptimizer,
    Inverted_Bananas,
    Inverted_Bananas_GsparseOptimizer,
    Inverted_Bananas_ZCP_GsparseOptimizer,
)
from naslib.optimizers.oneshot.gsparsity.shape_optimizer import GSparseOptimizer
from naslib.utils import get_zc_benchmark_api
from naslib import utils
from naslib.search_spaces import NasBench201SearchSpace, NasBench301SearchSpace
from naslib.utils import setup_logger, get_dataset_api, get_project_root, get_train_val_loaders
from naslib.search_spaces.core.query_metrics import Metric

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run NASLib optimizer with specified configuration."
)
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
    help="Optimizer type (rs, ls, bananas, drnas, gsparsity, zcp_gsparsity, inverted_bananas, inverted_bananas_gsparsity, inverted_bananas_zcp_gsparsity)",
)
# Add ZCP-specific arguments
parser.add_argument(
    "--zcp_method",
    type=str,
    default="jacov",
    help="Zero-cost predictor method (synflow, grad_norm, fisher, grasp, jacov, snip, flops, params)",
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
    help="Dataset (e.g., cifar10, cifar100, ImageNet16-120)",
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
parser.add_argument("--search_epochs", type=int, default=100, help="Number of search epochs")
parser.add_argument("--eval_epochs", type=int, default=600, help="Number of evaluation epochs")

args = parser.parse_args()

# Use parsed arguments
optimizer_type = args.optimizer
search_space_type = args.search_space
dataset = args.dataset
seed = args.seed
out_dir = args.out_dir
search_epochs = args.search_epochs
eval_epochs = args.eval_epochs
zcp_method = args.zcp_method

#! Multiply time that it takes for random search by a factor that captures the prunning rate that was used in the zcp-gs-nas and gs-nas methods, how to make it such that only the search of the second phase is pruned? Should I not take 3 zcp´s per condition but have it as HP that is automatically tuned as well? Thus able to have more compute to tune one methodoligy and probably better results.

#! Use the same random seeds for compared conditions, such that results are more comparable as they start more simililar e.g. param initialization, which images are sampled etc.

#! if I change search space to carry information of shapes than this is only the case for non querry optimizers and thus the comparability might suffer a bit. Also mention that I only do this to use the ZCP that are already implemented in NASLib. Else it would probably faster 

#! check for cifar10, 100 and ImageNet16-120

#! Please save my search trajectory, best model, best val acc and so on. Trainer.py should have options for this. Look at both uploaded files and decide, what the best approach would be. It should be able to resume from a checkpoint. Be aware of the interaction of HPO via Optuna and the training of the models them self.


#! TODO RAW DATA SPEICHERN, Irgend ein logging tool 

#! for any bo + any gs nas -> estimated bo time + actual oneshot time

# ! TPESampler() and HyperbandPruner(), maximize accuracy. For blackbox use querries for prunning; for oneshot use epochs for prunning, for commbined use querry + epoch estimated times; is it possible to start with default hpo setting and go from there?

#! Verify this shortly
# ! get automated script generation and running from history
# ! modified version of this for gsparsity and modified version of gsparsity

# TODO check if zcp bananas and inverted bananas are giving correct results and are truely using the zcps
# TODO check if zcp gsparsity is giving correct results
# TODO check if code for each method runs on nasbench201 cifar10, cifar100, ImageNet16-120 and on nasbench301 cifar10
# TODO does zcp_gsparsity use cuda?
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
    "epochs": eval_epochs,
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
            "epochs": search_epochs,
            "fidelity": -1,
        },
    },
    "ls": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": search_epochs,
            "num_init": 10,
        },
    },
    "bananas": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": search_epochs,  # ? #! https://github.com/naszilla/bananas/blob/main/nas_algorithms.py epochs = num_init + (total_queries - num_init) / kepochs = 10 + (150 - 10) / 10 = 24 -> to achieve 150 total architecture evaluations
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
    "zcp_bananas": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": search_epochs,  # ? #! https://github.com/naszilla/bananas/blob/main/nas_algorithms.py epochs = num_init + (total_queries - num_init) / kepochs = 10 + (150 - 10) / 10 = 24 -> to achieve 150 total architecture evaluations
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
            "zc": True,  # Enable zero-cost predictors
            "use_zc_api": True,
            "zc_names": [zcp_method],  # Should be a list
            "zc_only": True,  # Set to True if you want to use only ZC predictors
            "batch_size": 64, # threw error without
            "train_portion": 0.5 # threw error without
        },
    },
    "drnas": {
        "search": {  # ? https://github.com/cc-hpc-itwm/GSparsity/blob/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/drnas-c10-original/search-progressive-exp-20210227-085319/log.txt
            "checkpoint_freq": 5,
            "epochs": search_epochs,
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
            "epochs": search_epochs,  # in paper 100
            "grad_clip": 0,  # in paper 0
            "weight_decay": 60,  # original 120 in paper 60
            "threshold": 0.000001,
            "normalization": "div",  # in paper div
            "normalization_exponent": 0.5,  # in paper 0.5
            "learning_rate": 0.001,  # original 0.01 in paper 0.001
            "momentum": 0.8,  # in paper 0.8
            "learning_rate_min": 0.0001,  # in paper 0.0001
            "batch_size": 64,  # original 128; in log 64
            "train_portion": 0.95,  # originally 0.95 in paper 1
            "cutout": False,  # # in paper False (I think NASLIB only needs this for gsparsity on nasbench201)
            "cutout_length": 16,  # in paper 16 (I think NASLIB only needs this for gsparsity on nasbench201)
            # "cutout_prob": 1.0,  # (I think NASLIB only needs this for gsparsity on nasbench201)
        },
        #! zcp-bananas
        #! zcp_gsparsity
        #! bananas-gsparsity
        #! zcp-bananas-gsparsity

        #! Is there a straight forward way of combining inverted_bananas-gsparsity; zcp_inverted_bananas-gsparsity; inverted_bananas-zcp_gsparsity; zcp_inverted_bananas-zcp_gsparsity such that inverted_bananas is first applied to the search space, searches for the worst architectures, removes them and then zcp_gsparsity or gsparsity is applied reguraly to the remaining architectures?
    },
    "zcp_gsparsity": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": search_epochs,
            "grad_clip": 0,
            "weight_decay": 60,
            "threshold": 0.000001,
            "normalization": "div",
            "normalization_exponent": 0.5,
            "learning_rate": 0.001,
            "momentum": 0.8,
            "learning_rate_min": 0.0001,
            "batch_size": 64,
            "train_portion": 0.95,
            "cutout": False,
            "cutout_length": 16,
            "zcp_method": zcp_method,
        },
    },
    "inverted_bananas": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": search_epochs,  # ? #! https://github.com/naszilla/bananas/blob/main/nas_algorithms.py epochs = num_init + (total_queries - num_init) / kepochs = 10 + (150 - 10) / 10 = 24 -> to achieve 150 total architecture evaluations
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
    "inverted_bananas_gsparsity": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": search_epochs,
            "batch_size": 64,
            "train_portion": 0.5,
            "learning_rate": 0.001,
            "learning_rate_min": 0.0001,
            "momentum": 0.8,
        },
        # Stage 1 configuration (Inverted BANANAS)
        "stage1": {
            "epochs": search_epochs // 3,
            "k": 10,
            "num_init": 10,
            "num_ensemble": 5,
            "predictor_type": "mlp",
            "acq_fn_type": "its",
            "acq_fn_optimization": "mutation",
            "encoding_type": "path",
            "num_arches_to_mutate": 1,
            "max_mutations": 1,
            "num_candidates": 100,
            "removal_percentage": 0.3,
        },
        # Stage 2 configuration (GSparsity)
        "stage2": {
            "epochs": search_epochs * 2 // 3,
            "grad_clip": 0,
            "weight_decay": 60,
            "threshold": 0.000001,
            "normalization": "div",
            "normalization_exponent": 0.5,
            "learning_rate": 0.001,
            "momentum": 0.8,
            "learning_rate_min": 0.0001,
            "cutout": False,
            "cutout_length": 16,
        },
    },
    "inverted_bananas_zcp_gsparsity": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": search_epochs,
            "batch_size": 64,
            "train_portion": 0.5,
            "learning_rate": 0.001,
            "learning_rate_min": 0.0001,
            "momentum": 0.8,
        },
        # Stage 1 configuration (Inverted BANANAS)
        "stage1": {
            "epochs": search_epochs // 3,
            "k": 10,
            "num_init": 10,
            "num_ensemble": 5,
            "predictor_type": "mlp",
            "acq_fn_type": "its",
            "acq_fn_optimization": "mutation",
            "encoding_type": "path",
            "num_arches_to_mutate": 1,
            "max_mutations": 1,
            "num_candidates": 100,
            "removal_percentage": 0.3,
        },
        # Stage 2 configuration (ZCP GSparsity)
        "stage2": {
            "epochs": search_epochs * 2 // 3,
            "grad_clip": 0,
            "weight_decay": 60,
            "threshold": 0.000001,
            "normalization": "div",
            "normalization_exponent": 0.5,
            "learning_rate": 0.001,
            "momentum": 0.8,
            "learning_rate_min": 0.0001,
            "cutout": False,
            "cutout_length": 16,
            "zcp_method": zcp_method,
        },
    },
}

# Add common evaluation to all optimizer configs
for opt in optimizer_configs:
    optimizer_configs[opt]["evaluation"] = evaluation


def update_config(config, optimizer_type, search_space_type, dataset, seed, out_dir):
    """Update the configuration with experiment-specific settings"""
    # Set dataset name

    if optimizer_type == "gsparsity" or optimizer_type == "zcp_gsparsity" or optimizer_type == "inverted_bananas-gsparsity" or optimizer_type == "zcp_inverted_bananas-gsparsity" or optimizer_type == "drnas":
        config.save_arch_weights = False

    config.dataset = dataset

    config.data = str(get_project_root()) + "/data"  # path to naslib/data directory
    print(f"Data path: {config.data}")

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
        "nasbench201": {"cifar10": 10, "cifar100": 100, "ImageNet16-120": 120},
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
    elif optimizer_type == "zcp_gsparsity":
        optimizer = ZCP_GSparseOptimizer(config)
    elif optimizer_type == "inverted_bananas":
        optimizer = Inverted_Bananas(config)
    elif optimizer_type == "inverted_bananas_gsparsity":
        optimizer = Inverted_Bananas_GsparseOptimizer(config)
    elif optimizer_type == "inverted_bananas_zcp_gsparsity":
        optimizer = Inverted_Bananas_ZCP_GsparseOptimizer(config)
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Adapt the search space for the specific optimizer
    if optimizer_type == "drnas":
        optimizer.adapt_search_space(search_space=search_space, dataset=dataset)
    elif optimizer_type == "gsparsity":
        optimizer.adapt_search_space(search_space=search_space)
    elif optimizer_type == "zcp_gsparsity":
        train_loader, _, _, _, _ = get_train_val_loaders(config)
        optimizer.adapt_search_space(search_space=search_space, train_loader=train_loader)
    elif optimizer_type in ["rs", "ls", "bananas", "inverted_bananas"]:
        optimizer.adapt_search_space(search_space=search_space, dataset_api=dataset_api)
    elif optimizer_type in ["inverted_bananas_gsparsity", "inverted_bananas_zcp_gsparsity"]:
        # For two-stage optimizers
        train_loader = None
        if "zcp" in optimizer_type:
            train_loader, _, _, _, _ = get_train_val_loaders(config)
        optimizer.adapt_search_space(search_space=search_space, dataset_api=dataset_api, train_loader=train_loader)
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

        # Todo I want to save the search trajectory of each run
        # ! convert_naslib_nb201_to_str
        # Run the optimizer
        search_trajectory, best_model, best_val_acc = run_optimizer(
            optimizer_type, search_space_type, valid_dataset, config, s
        )

        print(f"Completed run for seed {s}")
        print(f"Best validation accuracy: {best_val_acc}")


if __name__ == "__main__":
    main()
