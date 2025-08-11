import logging
import os
from fvcore.common.config import CfgNode
import json
import argparse

import optuna
import optunahub
import copy
import time

import torch
import numpy as np

from naslib.optimizers import (
    RandomSearch,
    LocalSearch,
    Bananas,
    GSparseOptimizer,
    DrNASOptimizer,
    ZCP_GSparseOptimizer,
    Inverted_Bananas,
    Inverted_Bananas_GsparseOptimizer,
    Inverted_Bananas_ZCP_GsparseOptimizer,
)

#!!!!!!!
from naslib.optimizers.oneshot.gsparsity.zcp_minmax_gsparse_optimizer import (
    ZCP_GSparseOptimizer,
)

from naslib.optimizers.oneshot.gsparsity.zc_pre_reducing_search_space_Gsparse import (
    GSparseOptimizer as PreZCPGSparseOptimizer,
)
from naslib.optimizers.oneshot.gsparsity.zc_pre_reducing_search_space_zcp_minmax_gsparse_optimizer import (
    ZCP_GSparseOptimizer as PreZCPZCPGSparseOptimizer,
)

# Add imports for self-training optimizers
from naslib.optimizers.oneshot.gsparsity.self_training_bananas_optimizer import (
    Bananas as SelfTrainingBananas,
)
from naslib.optimizers.oneshot.gsparsity.self_training_inverted_bananas_optimizer import (
    Inverted_Bananas as SelfTrainingInvertedBananas,
)
from naslib.optimizers.oneshot.gsparsity.self_training_inverted_bananas_gsparse_optimizer import (
    Inverted_Bananas_GsparseOptimizer as SelfTrainingInvertedBananasGsparse,
)
from naslib.optimizers.oneshot.gsparsity.self_training_inverted_bananas_zcp_gsparse_optimizer import (
    Inverted_Bananas_ZCP_GsparseOptimizer as SelfTrainingInvertedBananasZCPGsparse,
)


from naslib import utils
from naslib.search_spaces import NasBench201SearchSpace, NasBench301SearchSpace
from naslib.utils import (
    setup_logger,
    get_dataset_api,
    get_project_root,
    get_train_val_loaders,
)
from naslib.search_spaces.core.query_metrics import Metric

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run NASLib optimizer with specified configuration."
)
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
    help="Optimizer type (rs, ls, bananas, drnas, gsparsity, zcp_gsparsity, inverted_bananas, inverted_bananas_gsparsity, inverted_bananas_zcp_gsparsity, random_sampling, self_training_bananas, self_training_inverted_bananas, self_training_inverted_bananas_gsparsity, self_training_inverted_bananas_zcp_gsparsity)",
)
# Add ZCP-specific arguments
parser.add_argument(
    "--zcp_method",
    type=str,
    default="jacov",
    help="Zero-cost predictor method (synflow, grad_norm, fisher, grasp, jacov, snip, nwot, epe_nas, zen, flops, params)",
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
parser.add_argument(
    "--search_epochs", type=int, default=100, help="Number of search epochs"
)
parser.add_argument(
    "--eval_epochs", type=int, default=600, help="Number of evaluation epochs"
)
parser.add_argument(
    "--resume",
    type=bool,
    default=False,
    help="Resume training from the last checkpoint. True, False",
)
# NEW: dataset subset as a CLI arg (was hardcoded before)
parser.add_argument(
    "--dataset_subset",
    type=float,
    default=0.2,
    help="Fraction of the dataset to use during search (0.0-1.0).",
)
# NEW: generic hyperparameter overrides
parser.add_argument(
    "--hp",
    "--set",
    dest="overrides",
    action="append",
    default=[],
    help="Override any hyperparameter using a dot path, e.g. "
    "--hp search.learning_rate=0.001 "
    "--hp evaluation.batch_size=128. "
    "Values are parsed as strings; numbers/bools/lists are auto-decoded by CfgNode.",
)

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
resume = args.resume

np.random.seed(seed)
torch.manual_seed(seed)

# Standard Query-based: comp_factor * scaling_epochs 200
# Query-based Self-Training: scaling_epochs / 200 * comp_factor
# Real-time Self-Training: comp_factor = 1; scaling_epochs = 1.0


# Maybe the problem is also related to the change of the logger in the configurator. As the statedict in the log says it is not complete thus the checkpoint after completing one epoch with gsparsity should also contain less. Or is this the case and my logging / printing information is just not nuanced enough to capture this?

# have look at log.log file


#! use computational factor for every time that is querried (multi shot and two stage)

#! maybe use one epoch (bias towards early influencial hp) / or much less data and more epochs for hpo than check hpo importance and with that reduce search space to imporatant params use the paper that says one epoch is unreasonably good and optuna

#! use logarithmic for left bounded [0,infinit]; [log a, log b] bischlHyperparameterOptimizationFoundations2021
#! should I use holdout/cross validation?

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

optimizer_configs = {
    "rs": {  #! has to be allowed to run longer to compensate for the hpo e.g. give runtime or maybe use time per epoch to calculate how many more epochs per dataset it should be allowed to run and check back if it went over the budget if it did remove till in budget
        "search": {
            "checkpoint_freq": 1,  #!
            "epochs": search_epochs,
            "fidelity": -1,
        },
    },
    "random_sampling": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": 1,
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
            "predictor_type": "mlp",  # ["bananas", "bayes_lin_reg", "bohamiann", "bonas", "dngo", "lgb", "gcn", "gp", "gpwl", "mlp", "nao", "ngb", "rf", "seminas", "sparse_gp", "var_sparse_gp", "xgb", "omni_ngb", "omni_seminas"]
            "acq_fn_type": "its",  # ["its", "ucb", "ei", "exploit_only"]
            "acq_fn_optimization": "mutation",  # ["mutation", "random_sampling"]
            "encoding_type": None,  #! not used predictor_type also specifies the encoding_type
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
            "zc": True,  #! Enable zero-cost predictors has to be true else it is no zcp_bananas
            "use_zc_api": True,
            "zc_names": [zcp_method],  # Should be a list
            "zc_only": True,  # Set to True if you want to use only ZC predictors
            "batch_size": 64,  # threw error without
            "train_portion": 0.5,  # threw error without
        },
    },
    "drnas": {  #! weight_decay (default: 0.0003), unrolled (default: False), arch_weight_decay (default: 0.001), op_optimizer (default: "SGD"), arch_optimizer (default: "Adam"), loss_criteria (default: "CrossEntropyLoss") are missing
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
    "zcp-pre_gsparsity": {  # ? https://github.com/cc-hpc-itwm/GSparsity/tree/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/gsparsity-c10/search-for-cell-lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195113 ; https://github.com/cc-hpc-itwm/GSparsity/blob/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/gsparsity-c10/scaling_div_0.5_accuracy_statistics.txt ; https://github.com/cc-hpc-itwm/GSparsity/blob/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/gsparsity-c10/search-for-cell-lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195113/_log_lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195113.txt ; plus paper
        "search": {
            "checkpoint_freq": 1,  #!
            "epochs": search_epochs,  # in paper 100
            "grad_clip": None,  # in paper 0
            "weight_decay": 60.0,  # original 120 in paper 60
            "threshold": 0.000001,
            "normalization": "div",  # in paper div
            "normalization_exponent": 0.5,  # in paper 0.5
            "learning_rate": 0.0001,  # original 0.01 in paper 0.001 #!!!!!!!!!!!!!!!!!!
            "momentum": 0.8,  # in paper 0.8
            "learning_rate_min": 0.0001,  # in paper 0.0001
            "batch_size": 64,  # original 128; in log 64
            "train_portion": 0.95,  # originally 0.95 in paper 1
            "cutout": False,  # # in paper False
            "cutout_length": 16,  # in paper 16
            "cutout_prob": 1.0,  # ADDED: allow override from HPO/final scripts
            "pre_computed_zc_scores": "naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor/arch_scores",
        },
    },
    "zcp-pre_zcp_gsparsity": {  # ? https://github.com/cc-hpc-itwm/GSparsity/tree/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/gsparsity-c10/search-for-cell-lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195113 ; https://github.com/cc-hpc-itwm/GSparsity/blob/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/gsparsity-c10/scaling_div_0.5_accuracy_statistics.txt ; https://github.com/cc-hpc-itwm/GSparsity/blob/d757f40be0178935aef705b9650002b7ed5f07ec/darts_space/logs/gsparsity-c10/search-for-cell-lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195113/_log_lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195113.txt ; plus paper
        "search": {
            "checkpoint_freq": 1,  #!
            "epochs": search_epochs,  # in paper 100
            "grad_clip": None,  # in paper 0
            "weight_decay": 60.0,  # original 120 in paper 60
            "threshold": 0.000001,
            "normalization": "div",  # in paper div
            "normalization_exponent": 0.5,  # in paper 0.5
            "learning_rate": 0.0001,  # original 0.01 in paper 0.001
            "momentum": 0.8,  # in paper 0.8
            "learning_rate_min": 0.0001,  # in paper 0.0001
            "batch_size": 64,  # original 128; in log 64
            "train_portion": 0.95,  # originally 0.95 in paper 1
            "cutout": False,
            "cutout_length": 16,
            "cutout_prob": 1.0,  # ADDED
            "zcp_method": zcp_method,  #! Enable zero-cost predictors
            "pre_computed_zc_scores": "naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor/arch_scores",
        },
    },
    "gsparsity": {
        "search": {
            "checkpoint_freq": 1,
            "epochs": search_epochs,
            "grad_clip": None,
            "weight_decay": 60.0,
            "threshold": 0.000001,
            "normalization": "div",
            "normalization_exponent": 0.5,
            "learning_rate": 0.0001,
            "momentum": 0.8,
            "learning_rate_min": 0.0001,
            "batch_size": 64,
            "train_portion": 0.95,
            "cutout": False,
            "cutout_length": 16,
            "cutout_prob": 1.0,
        },
    },
    "zcp_gsparsity": {
        "search": {
            "checkpoint_freq": 1,
            "epochs": search_epochs,
            "grad_clip": None,
            "weight_decay": 60.0,
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
            "cutout_prob": 1.0,
            "zcp_method": zcp_method,
        },
    },
    "inverted_bananas": {
        "search": {
            "checkpoint_freq": 1,  #!
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
            "checkpoint_freq": 1,  #!
            "epochs": search_epochs,
            "batch_size": 64,
            "train_portion": 0.5,
            "cutout": False,
            "cutout_length": 16,
            "cutout_prob": 1.0,  # ADDED
        },
        # Stage 1 configuration (Inverted BANANAS)
        "stage1": {
            "search": {
                # "epochs": search_epochs // 3,  #!
                "epochs": 3,
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
                # "removal_percentage": 0.3,
                "removal_percentage": 1.0,
            },
        },
        # Stage 2 configuration (GSparsity)
        "stage2": {
            "search": {
                # "epochs": search_epochs * 2 // 3,  #!
                "epochs": 5,
                "grad_clip": None,
                "weight_decay": 60.0,
                "threshold": 0.000001,
                "normalization": "div",
                "normalization_exponent": 0.5,
                "learning_rate": 0.0001,
                "momentum": 0.8,
                "learning_rate_min": 0.0001,
            },
        },
    },  #! problem when stage 1 complete but stage 2 epoch 0 not complete yet and I want to resume
    "inverted_bananas_zcp_gsparsity": {
        "search": {
            "checkpoint_freq": 1,  #!
            "epochs": search_epochs,
            "batch_size": 64,
            "train_portion": 0.5,
            "cutout": False,
            "cutout_length": 16,
            "cutout_prob": 1.0,  # ADDED
        },
        # Stage 1 configuration (Inverted BANANAS)
        "stage1": {
            "search": {
                # "epochs": search_epochs // 3, #!
                "epochs": 3,
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
                # "removal_percentage": 0.3,
                "removal_percentage": 1.0,
            },
        },
        # Stage 2 configuration (ZCP GSparsity)
        "stage2": {
            "search": {
                # "epochs": search_epochs // 3, #!
                "epochs": 3,
                "grad_clip": None,
                "weight_decay": 60.0,
                "threshold": 0.000001,
                "normalization": "div",
                "normalization_exponent": 0.5,
                "learning_rate": 0.001,
                "momentum": 0.8,
                "learning_rate_min": 0.0001,
                "zcp_method": zcp_method,
            },
        },
    },
    # Add new self-training optimizer configurations below
    "self_training_bananas": {
        "search": {
            "checkpoint_freq": 5,
            "epochs": search_epochs,
            "k": 10,
            "num_init": 10,
            "num_ensemble": 5,
            "predictor_type": "mlp",
            "acq_fn_type": "its",
            "acq_fn_optimization": "mutation",
            "encoding_type": None,
            "num_arches_to_mutate": 1,
            "max_mutations": 1,
            "num_candidates": 100,
            "train_epochs": 5,  #!
            "use_real_time": False,
        },
    },
    "self_training_inverted_bananas": {
        "search": {
            "checkpoint_freq": 1,
            "epochs": search_epochs,
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
            "train_epochs": 200,  #!
            "use_real_time": True,
        },
    },
    "self_training_inverted_bananas_gsparsity": {
        "search": {
            "checkpoint_freq": 1,
            "epochs": search_epochs,
            "batch_size": 64,
            "train_portion": 0.5,
            "cutout": False,
            "cutout_length": 16,
            "cutout_prob": 1.0,  # ADDED
            # "use_real_time": True, #!
            "use_real_time": False,
        },
        # Stage 1 configuration (Self-Training Inverted BANANAS)
        "stage1": {
            "search": {
                "epochs": 3,
                "train_epochs": 5,  #!
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
                "removal_percentage": 0.5,
            },
        },
        # Stage 2 configuration (GSparsity)
        "stage2": {
            "search": {
                "epochs": 3,
                "grad_clip": None,
                "weight_decay": 60.0,
                "threshold": 0.000001,
                "normalization": "div",
                "normalization_exponent": 0.5,
                "learning_rate": 0.001,
                "momentum": 0.8,
                "learning_rate_min": 0.0001,
            },
        },
    },
    "self_training_inverted_bananas_zcp_gsparsity": {
        "search": {
            "checkpoint_freq": 1,
            "epochs": search_epochs,
            "batch_size": 64,
            "train_portion": 0.5,
            "cutout": False,
            "cutout_length": 16,
            "cutout_prob": 1.0,  # ADDED
            "use_real_time": False,
        },
        # Stage 1 configuration (Self-Training Inverted BANANAS)
        "stage1": {
            "search": {
                "epochs": 3,
                "train_epochs": 3,
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
                "removal_percentage": 0.5,
            },
        },
        # Stage 2 configuration (ZCP GSparsity)
        "stage2": {
            "search": {
                "epochs": 3,
                "grad_clip": None,
                "weight_decay": 60.0,
                "threshold": 0.000001,
                "normalization": "div",
                "normalization_exponent": 0.5,
                "learning_rate": 0.001,
                "momentum": 0.8,
                "learning_rate_min": 0.0001,
                "zcp_method": zcp_method,
            },
        },
    },
}


# Add common evaluation to all optimizer configs
for opt in optimizer_configs:
    optimizer_configs[opt]["evaluation"] = evaluation

for opt in optimizer_configs:
    optimizer_configs[opt]["search"]["early_stopping"] = {
        "criterion": "valid_loss",  # Can be 'train_acc', 'train_loss', 'valid_acc', 'valid_loss', or 'runtime'
        "patience": 10,  # Number of epochs to wait for improvement
        "threshold": 0.00001,  # Minimum change to be considered an improvement
    }


def update_config(config, optimizer_type, search_space_type, dataset, seed, out_dir):
    """Update the configuration with experiment-specific settings"""
    # Set dataset name

    if (
        optimizer_type == "gsparsity"
        or optimizer_type == "zcp_gsparsity"
        or optimizer_type == "drnas"
        or optimizer_type == "inverted_bananas_gsparsity"
        or optimizer_type == "inverted_bananas_zcp_gsparsity"
        or optimizer_type == "self_training_inverted_bananas_zcp_gsparsity"
        or optimizer_type == "self_training_inverted_bananas_gsparsity"
        or optimizer_type == "zcp-pre_gsparsity"
        or optimizer_type == "zcp-pre_zcp_gsparsity"
    ):
        config.save_arch_weights = False

    is_self_training = "self_training" in optimizer_type

    if is_self_training:
        # Use the new computational factor for self-training methods
        comp_factor_path = os.path.join(
            "naslib/optimizers/oneshot/gsparsity/submission_scripts/self_training_bananas_verification_and_computational_factor/self_training_inverted_bananas/nasbench201",
            dataset,
            "results.json",
        )
    else:
        # Use the existing computational factor for query-based methods
        comp_factor_path = os.path.join(
            "naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_training_verification_and_querrybased_computational_factor",
            dataset,
            "results.json",
        )

    comp_factor = 1.0  # Default value
    if os.path.exists(comp_factor_path):
        try:
            with open(comp_factor_path, "r") as f:
                data = json.load(f)
                comp_factor = data["summary"]["part_time_computational_factor"][
                    "average"
                ]
            logging.info(
                f"Loaded computational factor {comp_factor} for {dataset} from {comp_factor_path}"
            )
        except Exception as e:
            logging.warning(
                f"Warning: Could not load computational factor from {comp_factor_path}. Using default {comp_factor}. Error: {e}"
            )
    else:
        logging.warning(
            f"Warning: Computational factor file not found at {comp_factor_path}. Using default {comp_factor}."
        )
    config.search.comp_factor = comp_factor

    # Set the number of epochs to scale queried time by.
    if search_space_type == "nasbench201":
        if is_self_training and not getattr(config.search, "use_real_time", False):
            # Self-training that queries the benchmark.
            # The benchmark returns time for 200 epochs. We scale it to the desired `train_epochs`.
            train_epochs = (
                config.stage1.search.train_epochs
                if "stage1" in config
                else config.search.train_epochs
            )
            config.search.scaling_factor_epochs = train_epochs
            logging.info(
                f"Self-training: Scaling NB201 time by train_epochs = {train_epochs}."
            )
        elif not is_self_training:
            # Standard query-based methods. We want the full 200-epoch time.
            config.search.scaling_factor_epochs = 200
            logging.info(f"Query-based: Scaling NB201 time by fixed 200 epochs.")
        else:
            # This covers self-training with use_real_time=True
            config.search.scaling_factor_epochs = 1.0
            logging.info(
                "Self-training (real-time): Using measured time. Scaling factor is 1.0."
            )
    else:
        config.search.scaling_factor_epochs = 1  # Default for other search spaces

    # If use_real_time is set, we use the actual measured time from self-training optimizers.
    # The trainer multiplies the returned time by these factors, so we set them to 1.
    if getattr(config.search, "use_real_time", False):
        config.search.comp_factor = 1.0
        config.search.scaling_factor_epochs = 1.0
        logging.info(
            f"Optimizer '{optimizer_type}' has 'use_real_time' set. "
            f"Setting comp_factor and scaling_factor_epochs to 1.0 to use real runtime."
        )

    config.dataset = dataset
    # CHANGED: allow overriding dataset subset via CLI
    config.dataset_subset = args.dataset_subset

    config.data = str(get_project_root()) + "/data"  # path to naslib/data directory
    print(f"Data path: {config.data}")

    # Set search space
    config.search_space = search_space_type

    # Set up output directory path
    if "inverted_bananas_zcp_gsparsity" in optimizer_type:
        config.save = f"{out_dir}/{optimizer_type}/{config.stage2.search.zcp_method}/{search_space_type}/{dataset}/{seed}"
    elif "zcp_gsparsity" in optimizer_type:
        config.save = f"{out_dir}/{optimizer_type}/{config.search.zcp_method}/{search_space_type}/{dataset}/{seed}"
    else:
        config.save = f"{out_dir}/{optimizer_type}/{search_space_type}/{dataset}/{seed}"

    # Set seed
    config.search.seed = seed

    return config


def _parse_hp_overrides(overrides_list):
    """
    Convert repeated --hp key=value entries into a list suitable for CfgNode.merge_from_list.
    Example: ['search.learning_rate=0.01', 'evaluation.batch_size=128']
    -> ['search.learning_rate', '0.01', 'evaluation.batch_size', '128']
    """
    merged = []
    for entry in overrides_list:
        if "=" not in entry:
            raise ValueError(f"Invalid --hp format (missing '='): {entry}")
        key, val = entry.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            raise ValueError(f"Invalid --hp key: {entry}")
        merged.extend([key, val])
    return merged


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

    # Configure the 'fvcore' logger to use the same file handler
    # as the main application logger.
    app_file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            app_file_handler = handler
            break

    if app_file_handler:
        fvcore_logger = logging.getLogger("fvcore")
        # Add the application's file handler to the fvcore logger
        fvcore_logger.addHandler(app_file_handler)
        # Set the level for the fvcore logger. INFO will capture INFO and WARNING messages.
        fvcore_logger.setLevel(logging.INFO)
        # Prevent fvcore messages from being propagated to ancestor loggers,
        # as they are now explicitly handled by the app_file_handler.
        # This helps avoid duplicate messages if the root logger also has handlers (e.g., console).
        fvcore_logger.propagate = False
    else:
        # Log a warning if the file handler couldn't be found, as fvcore logs might not be saved.
        logger.warning(
            "Could not find FileHandler for the main logger. "
            "Fvcore logs might not be written to the log file."
        )

    # Log the configuration
    logger.info(f"Configuration is \n{config}")

    # Set up the seed
    utils.set_seed(seed)

    search_resume_from = ""
    eval_resume_from = ""

    if resume:
        search_resume_from = utils.get_last_checkpoint(config, search=True)
        logger.info(
            f"RunOptimizer: utils.get_last_checkpoint(search=True) returned: '{search_resume_from}'"
        )
        eval_resume_from = utils.get_last_checkpoint(config, search=False)
        logger.info(
            f"RunOptimizer: utils.get_last_checkpoint(search=False) returned: '{eval_resume_from}'"
        )
        if search_resume_from:
            logger.info(
                f"RunOptimizer: Attempting to resume search from checkpoint: {search_resume_from}"
            )
        if eval_resume_from:
            logger.info(
                f"RunOptimizer: Attempting to resume evaluation from checkpoint: {eval_resume_from}"
            )
    else:
        logger.info("RunOptimizer: Not resuming (config_node_arg.resume is False).")

    logger.info(
        f"RunOptimizer: Passing resume_from='{search_resume_from}' to trainer.search()"
    )
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
    if optimizer_type == "rs" or optimizer_type == "random_sampling":
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
    elif optimizer_type == "self_training_bananas":
        optimizer = SelfTrainingBananas(config)
    elif optimizer_type == "self_training_inverted_bananas":
        optimizer = SelfTrainingInvertedBananas(config)
    elif optimizer_type == "self_training_inverted_bananas_gsparsity":
        optimizer = SelfTrainingInvertedBananasGsparse(config)
    elif optimizer_type == "self_training_inverted_bananas_zcp_gsparsity":
        optimizer = SelfTrainingInvertedBananasZCPGsparse(config)
    elif optimizer_type == "zcp-pre_gsparsity":
        optimizer = PreZCPGSparseOptimizer(config)
    elif optimizer_type == "zcp-pre_zcp_gsparsity":
        optimizer = PreZCPZCPGSparseOptimizer(config)
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Adapt the search space for the specific optimizer
    if optimizer_type == "drnas":
        optimizer.adapt_search_space(search_space=search_space, dataset=dataset)
    elif optimizer_type == "gsparsity":
        optimizer.adapt_search_space(search_space=search_space)
    elif optimizer_type == "zcp_gsparsity":
        train_loader, _, _, _, _ = get_train_val_loaders(
            config, train_workers=0, val_workers=0
        )
        optimizer.adapt_search_space(
            search_space=search_space, train_loader=train_loader
        )
    elif (
        optimizer_type == "zcp-pre_gsparsity"
        or optimizer_type == "zcp-pre_zcp_gsparsity"
    ):
        train_loader, _, _, _, _ = get_train_val_loaders(
            config, train_workers=0, val_workers=0
        )
        optimizer.adapt_search_space(
            search_space=search_space,
            train_loader=train_loader,
            resume_from_path=search_resume_from,
        )
    elif optimizer_type in [
        "rs",
        "ls",
        "bananas",
        "inverted_bananas",
        "random_sampling",
        "self_training_bananas",
        "self_training_inverted_bananas",
    ]:
        optimizer.adapt_search_space(search_space=search_space, dataset_api=dataset_api)
    elif optimizer_type in [
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
        "self_training_inverted_bananas_gsparsity",
        "self_training_inverted_bananas_zcp_gsparsity",
    ]:
        # For two-stage optimizers
        train_loader = None
        if "zcp" in optimizer_type:
            train_loader, _, _, _, _ = get_train_val_loaders(
                config, train_workers=0, val_workers=0
            )
        optimizer.adapt_search_space(
            search_space=search_space,
            dataset_api=dataset_api,
            train_loader=train_loader,
        )
    else:
        optimizer.adapt_search_space(search_space=search_space)

    # Create trainer and run search
    if optimizer_type in [
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
        "self_training_inverted_bananas_gsparsity",
        "self_training_inverted_bananas_zcp_gsparsity",
    ]:
        # from naslib.defaults.two_stage_trainer import Trainer
        from naslib.defaults.two_stage_trainer_multi_dataloading_workers import Trainer

        trainer = Trainer(optimizer, config, lightweight_output=False)
    else:
        # from naslib.defaults.trainer import Trainer
        from naslib.defaults.trainer_multi_dataloading_workers import Trainer

        trainer = Trainer(optimizer, config, lightweight_output=False)

    trainer.search(resume_from=search_resume_from, report_incumbent=False)

    # Get the search trajectory
    search_trajectory = trainer.search_trajectory
    logger.info(f"Train accuracies: {search_trajectory.train_acc}")
    logger.info(f"Validation accuracies: {search_trajectory.valid_acc}")

    # Evaluate the best model found in search
    # The search_model argument in evaluate will default to 'model_final.pth'
    # from the search directory if not provided, which is suitable after a search.
    best_model_val_acc = trainer.evaluate(
        dataset_api=dataset_api,
        metric=Metric.VAL_ACCURACY,
        resume_from=eval_resume_from,
    )
    logger.info(f"Best model validation accuracy: {best_model_val_acc}")

    # Get the best model architectureP
    # best_model = optimizer.get_final_architecture()

    # return search_trajectory, best_model, best_model_val_acc
    return search_trajectory, best_model_val_acc


def main():
    """Main function to run the optimizer"""
    # Create base configuration from the selected optimizer
    if optimizer_type in optimizer_configs:
        config = optimizer_configs[optimizer_type]
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Convert dictionary to CfgNode
    config = CfgNode.load_cfg(json.dumps(config))

    # NEW: apply generic hyperparameter overrides BEFORE update_config
    if args.overrides:
        try:
            override_list = _parse_hp_overrides(args.overrides)
            # Allow adding new keys only if they already exist; CfgNode will error on unknown keys.
            config.merge_from_list(override_list)
            print(f"Applied overrides: {args.overrides}")
        except Exception as e:
            raise ValueError(f"Failed to apply --hp overrides: {e}")

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
        search_trajectory, best_val_acc = run_optimizer(
            optimizer_type, search_space_type, valid_dataset, config, s
        )

        print(f"Completed run for seed {s}")
        print(f"Best validation accuracy: {best_val_acc}")


if __name__ == "__main__":
    main()
