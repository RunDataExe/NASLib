import logging
import os
from fvcore.common.config import CfgNode
import json
import argparse
import numpy as np
import optuna
import optunahub
import copy
import time
import torch
import shutil


# DONE and verified till here
# TODO 12,4 workers
# TODO implement gsparsity and zcp error json handling like in two stage
# TODO verify time results for one-shot, multishot and two-stage methods for each dataset (configurator)
# TODO use Accuracy again as metric for the objective function
# TODO comp factor should only count training time as training not the time it took for test
# TODO verify hpo for one stage and two stage (accuracy/objective and runtimes)
# ? DONE but not verified
# TODO rerun computational factor
# TODO verify self_training_inverted_bananas_gsparsity imporant batchsize etc for first stage with self training !!!
# TODO verfify search space (bananas ensemble options)
# TODO verify resumption of non self training inverted bananas gsparsity
# TODO verify resumption of non self training inverted bananas zcp gsparsity
# TODO check the zcp scores for jacov, param and synflow for 0 values (jacov seems to do)
# TODO create and verify self training inverted bananas zcp gsparsity
# TODO use my own self_training versions for wide and narrow hpo
# TODO check the conditions of the predictors for BANANAS some might only work if I use ZCP which I will not do
# TODO check that HPO space is the same for all methods e.g. gsparsity, zcp_gsparsity, second stage gsparsity, second stage zcp_gsparsity
# TODO check if HPO space makes sense and change it if necessary
# TODO make use of sqlite db to store the results of the trials
# TODO Hyperparameter Importance
# TODO also make comp factor for 0 workers such that I have fair comp for the self train bananas stuff (e.g. scale it up to the speed it would have with 12,4 workers)
# TODO verify self_training_bananas
# TODO verify self_training_inverted_bananas -> 200 epochs training vs querry based inverted bananas
# TODO enable correct usage of computational factor for first stage of self_training_inverted_bananas_gsparsity
# TODO verify time for self_training_inverted_bananas_gsparsity
# TODO use computational factor for first stage of self_training_inverted_bananas_zcp_gsparsity
# TODO Filter db to remove trials that would have not been permitted by the budget and create a new study with the filtered trials that I will use for importance checking and visualization
# TODO check if log.log for selftraining_inverted and inverted gives the same
# TODO enable that new trials can be added to a study if the same setting is reran (check if runtime would exceed / is exeeded, before you allow (trials naslib))
# TODO optuna visualization
# TODO verify time for self_training_inverted_bananas_ methods in logs/errors.json
# TODO make decision about zcp size transformation (currently focus on middle of image despite random)
# TODO make decision about Early Stopping prob will go for validation loss as it is continous and encapsulates how much the model is off the target (could also run very long and choose the best based on checkpoints afterwards but how will i treat self training ibo? I normaly would have to tune patience and threshold but with DEHB I already have some sort of early stopping thus tuning it is probably not really informative)
# TODO decide where everywhere to use early stopping (also in selflearning inverted bananas?)
# TODO make wide but other budget script
# TODO decide epochs of first stage and internal epochs and than start hpo for this

# * OPEN
#! Today:
#! TODO the two stage methods do not resume the pruning when 1 stage fails and is resumed
# TODO set the correct reduction factor etc
# TODO check resumption / adding trials with new seeding (gsparsity and selftraining_ibo_gsparsity)
# Todo start iBO GSparsity HPO (CIFAR100)
# TODO start Gsparsity HPO (CIFAR100)

# TODO sum of l2 norm as normalization of the scores

# TODO make decision about ZCP normalization
# TODO make decision about Dimensionality of Synthetic network
#! TODO start zcp hpo as fast as possible

# Later if hpo runs are running / working
# TODO order seeds such that they are in one row instead of vertically ordered
# TODO performance incumbent acc, stability raw acc
# TODO cumulative AUC ylabel = Cumulative Incumbent Accuracy (%·s) [Linear Scale]; Cumulative Raw Accuracy (%·s) [Linear Scale]
# TODO performance stability AUC ylabel = Incumbent Accuracy (%) [Linear Scale]; Raw Accuracy (%) [Linear Scale]
# TODO titels Anytime Performance DATASET BENCHMARK; Anytime Stability DATASET BENCHMARK; Cumulative Performance DATASET BENCHMARK; Cumulative Stability DATASET BENCHMARK (that final hp and how auc is made and seed num etc in footnote in the paper itself)
# TODO make Final Performance Plot / AUC / cumulative AUC
# TODO make Final Stability Plot / AUC / cumulative AUC
# TODO validate the auc calculation
# TODO implement normalized AUC to make comparison easier
# TODO maybe only if I realy have time make generalization plots across all datasets per method

# TODO script that takes the best hp per method / dataset combination and creates slurm scripts for each run e.g. arg configs for each method


# TODO check epochs that the methods run in WHPO
# TODO check hpo such that the studies are not pruned for wide search space or that the importance analysis is using pruned and completed for this part of the study

# TODO save into lazygit and proceed with other tasks
# TODO make Final Training Plot / AUC / cumulative AUC
# TODO make first test plots to show in presentation
# TODO make presentation slides


# TODO implement early stopping for the methods such that they are trained till convergence not till fixed point

# TODO enable seed tuning

# Would be even better if I ran BANANAS for real in the HPO using nasbenchs training hp for 1 epoch to get a fairer hpo signal and I would not have the problem of unfair hyper band prunning
# this tries to avoid the oracle problem in which the ibo is probably a lot more influencial in the hp than the oneshots hpo
# also saves the comp budget for the first stage in the final setting which is a huge deal
# 1. HPO Phase (Wide & Narrow):
# Action: Use a fair, noisy signal for all methods.
# One-Shot: Real validation accuracy after a few epochs.
# Two-Stage (BANANAS): Query the benchmark for early-epoch (e.g., 12-epoch) accuracy. This is fast but provides a noisy signal comparable to the one-shot method.
# Outcome: The HPO is fair. It finds the best hyperparameters for algorithms working with realistic, limited information.
# 2. Final Comparison Run (Multiple Seeds):
# One-Stage Method (e.g., GSparsity):
# Time: Measure the real wall-clock time for its 100-epoch search.
# Performance: Query the 200-epoch accuracy of the final architecture.
# Two-Stage Method (BANANAS + GSparsity):
# Time (Stage 1): Sum the train_time for all 200-epoch queries and multiply by your computational_factor.
# Time (Stage 2): Measure the real wall-clock time for its 100-epoch search.
# Total Time: (Simulated Time from Stage 1) + (Real Time from Stage 2).
# Performance: Query the 200-epoch accuracy of the final architecture.
# Baseline (Random Search):
# Time: Sum the train_time for all 200-epoch queries and multiply by your computational_factor.
# Performance: The 200-epoch accuracy of the best architecture found.

# Run the search algorithm for its budgeted duration.
# For GSparsity, this means running the search for 1 epoch (or your defined HPO budget).
# For Inverted BANANAS + GSparsity, this means running stage 1 for 1 epoch and stage 2 for 1 epoch.
# Get the final, discretized architecture that the algorithm produces at the end of its run (optimizer.get_final_architecture()).
# Query the benchmark for the 200-epoch validation accuracy of that specific final architecture.
# Return this queried accuracy as the value for Optuna to maximize.

# TODO save into lazygit and proceed with other tasks
# TODO test hpo
# TODO make a setting for one method each on cifar10 with in narrow search with seed as hpo and check its influence on the final arch
# TODO save into lazygit and proceed with other tasks
# TODO test hpo
# TODO save into lazygit and proceed with other tasks
# TODO test Plotting for Gsparsity / ibogsnas (no zcp approach)

# // TODO check validation batches into list
# //batch_size: 64
# //train_portion: 0.9466741536616483
# //42 validation batches
# //batch_size: 64
# //train_portion: 0.856805511717202
# //112 validation batches
# ? Maybe also plot the loss lines to check for overfitting and so on


# Also set the number of threads for PyTorch to 1
# torch.set_num_threads(16)
# # Add this to print PyTorch config for debugging
# print("--- PyTorch Configuration ---")
# # torch.show_config() was added in torch 1.7. Use the underlying call for compatibility.
# print(torch.__config__.show())
# print("-----------------------------")
# import multiprocessing as mp

from optuna.trial import TrialState, FrozenTrial

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

from naslib.optimizers.oneshot.gsparsity.zcp_minmax_gsparse_optimizer import (
    ZCP_GSparseOptimizer as ZCP_GSparseOptimizer,
)

from naslib.optimizers.oneshot.gsparsity.self_training_inverted_bananas_gsparse_optimizer import (
    Inverted_Bananas_GsparseOptimizer as SelfTrainingInvertedBananasGsparse,
)
from naslib.optimizers.oneshot.gsparsity.self_training_inverted_bananas_zcp_gsparse_optimizer import (
    Inverted_Bananas_ZCP_GsparseOptimizer as SelfTrainingInvertedBananasZCPGsparse,
)

from naslib.optimizers.oneshot.gsparsity.zc_pre_reducing_search_space_Gsparse import (
    GSparseOptimizer as PreZCPGSParseOptimizer,
)
from naslib.optimizers.oneshot.gsparsity.zc_pre_reducing_search_space_zcp_minmax_gsparse_optimizer import (
    ZCP_GSparseOptimizer as PreZCPZCPGSparseOptimizer,
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
    "--hpo_timeout",
    type=int,
    default=None,
    help="Timeout in seconds for the HPO study.",
)
parser.add_argument(
    "--resume",
    type=bool,
    default=False,
    help="Resume training from the last checkpoint. True, False",
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
hpo_timeout = args.hpo_timeout


#!!!
np.random.seed(seed)
torch.manual_seed(seed)


# Use these functions to decide which trials to retry or skip
# ...existing code...

# Maybe the problem is also related to the change of the logger in the configurator. As the statedict in the log says it is not complete thus the checkpoint after completing one epoch with gsparsity should also contain less. Or is this the case and my logging / printing information is just not nuanced enough to capture this?

# have look at log.log file


#! update logging to capture the hp settings

#! check standard value ranges for each hyperparameter (papers for resnet and conv nets)

#! use computational factor for every time that is querried (multi shot and two stage)

#! maybe use one epoch (bias towards early influencial hp) / or much less data and more epochs for hpo than check hpo importance and with that reduce search space to imporatant params use the paper that says one epoch is unreasonably good and optuna

#! use logarithmic for left bounded [0,infinit]; [log a, log b] bischlHyperparameterOptimizationFoundations2021
#! should I use holdout/cross validation?

#! Multiply time that it takes for random search by a factor that captures the prunning rate that was used in the zcpgs-nas and gs-nas methods, how to make it such that only the search of the second phase is pruned? Should I not take 3 zcp´s per condition but have it as HP that is automatically tuned as well? Thus able to have more compute to tune one methodoligy and probably better results.

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

#! maybe increase lr range
# trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)


def _clear_logging_handlers():
    """
    Removes all handlers from the root, naslib, and fvcore loggers
    to prevent duplicate logging messages in consecutive Optuna trials.
    """
    loggers_to_clear = [
        logging.getLogger(),  # Root logger
        logging.getLogger("naslib"),
        logging.getLogger("fvcore"),
    ]

    for logger in loggers_to_clear:
        if logger.hasHandlers():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


logging.getLogger("sampler").setLevel(logging.INFO)

import hashlib


def trial_hash(hp_config):
    """
    Returns a deterministic hash for a hyperparameter configuration.
    """
    # Serialize config as a sorted JSON string
    config_str = json.dumps(hp_config, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()


def is_trial_run(hp_config, run_hashes):
    """
    Checks if the trial (hp_config) has already been run.
    run_hashes: set of hashes of already run configs
    """
    return trial_hash(hp_config) in run_hashes


# import pprint


# def debug_trial_params(trial_a, trial_b):
#     print("Trial A params:")
#     pprint.pprint(trial_a.params)
#     print("Trial B params:")
#     pprint.pprint(trial_b.params)
#     print("Trial A types:")
#     pprint.pprint({k: type(v) for k, v in trial_a.params.items()})
#     print("Trial B types:")
#     pprint.pprint({k: type(v) for k, v in trial_b.params.items()})
#     print("Are dicts equal?", trial_a.params == trial_b.params)


def _copy_trial_artifacts(source_trial_number, dest_trial_path, base_out_dir):
    """Copies all artifacts from a source trial directory to a destination."""
    # Reconstruct the path to the source trial's directory
    # Note: This assumes the directory structure defined in `update_config`
    source_trial_path = os.path.dirname(dest_trial_path)  # Gets the parent directory
    source_trial_path = os.path.join(source_trial_path, f"trial_{source_trial_number}")

    if os.path.isdir(source_trial_path):
        logging.info(f"Copying artifacts from {source_trial_path} to {dest_trial_path}")
        # Ensure destination exists and is empty to avoid merging issues
        if os.path.exists(dest_trial_path):
            shutil.rmtree(dest_trial_path)
        os.makedirs(dest_trial_path, exist_ok=True)
        shutil.copytree(source_trial_path, dest_trial_path, dirs_exist_ok=True)
    else:
        logging.warning(
            f"Source directory for artifact copy not found: {source_trial_path}. "
            f"New trial directory will be empty."
        )


def objective(trial):
    # If not pruned above, this is a new trial or the last failed trial to retry
    config = {}

    if optimizer_type == "gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "grad_clip": trial.suggest_categorical(
                    "grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 30.0, 150.0, log=True
                ),
                "threshold": None,  # Not used in the current implementation of GSparsity
                # "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                "normalization": trial.suggest_categorical(
                    "normalization", ["none", "mul", "div"]
                ),
                "normalization_exponent": trial.suggest_float(
                    "normalization_exponent", 0.25, 0.75
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-1, log=True
                ),
                "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                "learning_rate_min": trial.suggest_float(
                    "learning_rate_min", 1e-5, 5e-4, log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "train_portion": trial.suggest_float("train_portion", 0.8, 0.99),
            },
        }
    elif optimizer_type == "zcp_gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "grad_clip": trial.suggest_categorical(
                    "grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 30.0, 150.0, log=True
                ),
                "threshold": None,  # Not used in the current implementation of GSparsity
                # "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                "normalization": trial.suggest_categorical(
                    "normalization", ["none", "mul", "div"]
                ),
                "normalization_exponent": trial.suggest_float(
                    "normalization_exponent", 0.25, 0.75
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-1, log=True
                ),
                "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                "learning_rate_min": trial.suggest_float(
                    "learning_rate_min", 1e-5, 5e-4, log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "train_portion": trial.suggest_float("train_portion", 0.8, 0.99),
                "zcp_method": zcp_method,
            },
        }
    elif optimizer_type == "zcp-pre_gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "grad_clip": trial.suggest_categorical(
                    "grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 30.0, 150.0, log=True
                ),
                "threshold": None,  # Not used in the current implementation of GSparsity
                # "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                "normalization": trial.suggest_categorical(
                    "normalization", ["none", "mul", "div"]
                ),
                "normalization_exponent": trial.suggest_float(
                    "normalization_exponent", 0.25, 0.75
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-1, log=True
                ),
                "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                "learning_rate_min": trial.suggest_float(
                    "learning_rate_min", 1e-5, 5e-4, log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "train_portion": trial.suggest_float("train_portion", 0.8, 0.99),
                "pre_computed_zc_scores": "naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor/arch_scores",
            },
        }
    elif optimizer_type == "zcp-pre_zcp_gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "grad_clip": trial.suggest_categorical(
                    "grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 30.0, 150.0, log=True
                ),
                "threshold": None,  # Not used in the current implementation of GSparsity
                # "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                "normalization": trial.suggest_categorical(
                    "normalization", ["none", "mul", "div"]
                ),
                "normalization_exponent": trial.suggest_float(
                    "normalization_exponent", 0.25, 0.75
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-1, log=True
                ),
                "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                "learning_rate_min": trial.suggest_float(
                    "learning_rate_min", 1e-5, 5e-4, log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "train_portion": trial.suggest_float("train_portion", 0.8, 0.99),
                "zcp_method": zcp_method,
                "pre_computed_zc_scores": "naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor/arch_scores",
            },
        }
    elif optimizer_type == "inverted_bananas_gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "train_portion": trial.suggest_float("train_portion", 0.8, 0.99),
            },
            # Stage 1 configuration (Inverted BANANAS)
            "stage1": {
                "search": {
                    "epochs": search_epochs // 2,
                    "k": trial.suggest_int("k", 5, 20),
                    "num_init": trial.suggest_int("num_init", 5, 20),
                    "num_ensemble": trial.suggest_int("num_ensemble", 3, 7),
                    "predictor_type": trial.suggest_categorical(
                        "predictor_type",
                        ["mlp", "lgb", "xgb", "rf", "ngb", "gcn", "gp"],
                    ),
                    "acq_fn_type": trial.suggest_categorical(
                        "acq_fn_type", ["its", "ucb", "ei"]
                    ),
                    "encoding_type": None,  # is useless as its set by the predictor type
                    "num_candidates": trial.suggest_int("num_candidates", 50, 200),
                    "removal_percentage": 0.25,
                },
            },
            # Stage 2 configuration (GSparsity)
            "stage2": {
                "search": {
                    "epochs": search_epochs // 2,
                    "grad_clip": trial.suggest_categorical(
                        "grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                    ),
                    "weight_decay": trial.suggest_float(
                        "weight_decay", 30.0, 150.0, log=True
                    ),
                    "threshold": None,  # Not used in the current implementation of GSparsity
                    # "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                    "normalization": trial.suggest_categorical(
                        "normalization", ["none", "mul", "div"]
                    ),
                    "normalization_exponent": trial.suggest_float(
                        "normalization_exponent", 0.25, 0.75
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-4, 1e-1, log=True
                    ),
                    "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                    "learning_rate_min": trial.suggest_float(
                        "learning_rate_min", 1e-5, 5e-4, log=True
                    ),
                },
            },
        }
    elif optimizer_type == "inverted_bananas_zcp_gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "train_portion": trial.suggest_float("train_portion", 0.8, 0.99),
            },
            # Stage 1 configuration (Inverted BANANAS)
            "stage1": {
                "search": {
                    "epochs": search_epochs // 2,
                    "k": trial.suggest_int("k", 5, 20),
                    "num_init": trial.suggest_int("num_init", 5, 20),
                    "num_ensemble": trial.suggest_int("num_ensemble", 3, 7),
                    "predictor_type": trial.suggest_categorical(
                        "predictor_type",
                        ["mlp", "lgb", "xgb", "rf", "ngb", "gcn", "gp"],
                    ),
                    "acq_fn_type": trial.suggest_categorical(
                        "acq_fn_type", ["its", "ucb", "ei"]
                    ),
                    "encoding_type": None,  # is useless as its set by the predictor type
                    "num_candidates": trial.suggest_int("num_candidates", 50, 200),
                    "removal_percentage": 0.25,
                },
            },
            # Stage 2 configuration (ZCP GSparsity)
            "stage2": {
                "search": {
                    "epochs": search_epochs // 2,
                    "grad_clip": trial.suggest_categorical(
                        "grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                    ),
                    "weight_decay": trial.suggest_float(
                        "weight_decay", 30.0, 150.0, log=True
                    ),
                    "threshold": None,  # Not used in the current implementation of GSparsity
                    # "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                    "normalization": trial.suggest_categorical(
                        "normalization", ["none", "mul", "div"]
                    ),
                    "normalization_exponent": trial.suggest_float(
                        "normalization_exponent", 0.25, 0.75
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-4, 1e-1, log=True
                    ),
                    "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                    "learning_rate_min": trial.suggest_float(
                        "learning_rate_min", 1e-5, 5e-4, log=True
                    ),
                    "zcp_method": zcp_method,
                },
            },
        }
    elif optimizer_type == "self_training_inverted_bananas_gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "train_portion": trial.suggest_float("train_portion", 0.8, 0.99),
                "use_real_time": False,
            },
            # Stage 1 configuration (Self-Training Inverted BANANAS)
            "stage1": {
                "search": {
                    # "epochs": search_epochs // 2,
                    "epochs": 1,  # For HPO, we only train for 1 epoch
                    "train_epochs": 1,  # For HPO, we only train for 1 epoch
                    "k": trial.suggest_int("k", 5, 20),
                    "num_init": trial.suggest_int("num_init", 5, 20),
                    "num_ensemble": trial.suggest_int("num_ensemble", 3, 7),
                    "predictor_type": trial.suggest_categorical(
                        "predictor_type",
                        ["mlp", "lgb", "xgb", "rf", "ngb", "gcn", "gp"],
                    ),
                    "acq_fn_type": trial.suggest_categorical(
                        "acq_fn_type", ["its", "ucb", "ei"]
                    ),
                    "encoding_type": None,
                    "num_candidates": trial.suggest_int("num_candidates", 50, 200),
                    "removal_percentage": 0.25,
                },
            },
            # Stage 2 configuration (GSparsity)
            "stage2": {
                "search": {
                    "epochs": 1,
                    "grad_clip": trial.suggest_categorical(
                        "grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                    ),
                    "weight_decay": trial.suggest_float(
                        "weight_decay", 30.0, 150.0, log=True
                    ),
                    "threshold": None,  # Not used in the current implementation of GSparsity
                    # "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                    "normalization": trial.suggest_categorical(
                        "normalization", ["none", "mul", "div"]
                    ),
                    "normalization_exponent": trial.suggest_float(
                        "normalization_exponent", 0.25, 0.75
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-4, 1e-1, log=True
                    ),
                    "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                    "learning_rate_min": trial.suggest_float(
                        "learning_rate_min", 1e-5, 5e-4, log=True
                    ),
                },
            },
        }
    elif optimizer_type == "self_training_inverted_bananas_zcp_gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "train_portion": trial.suggest_float("train_portion", 0.8, 0.99),
                "use_real_time": False,
            },
            # Stage 1 configuration (Self-Training Inverted BANANAS)
            "stage1": {
                "search": {
                    # "epochs": search_epochs // 2,
                    "epochs": 1,  # For HPO, we only train for 1 epoch
                    "train_epochs": 1,  # For HPO, we only train for
                    "k": trial.suggest_int("k", 5, 20),
                    "num_init": trial.suggest_int("num_init", 5, 20),
                    "num_ensemble": trial.suggest_int("num_ensemble", 3, 7),
                    "predictor_type": trial.suggest_categorical(
                        "predictor_type",
                        ["mlp", "lgb", "xgb", "rf", "ngb", "gcn", "gp"],
                    ),
                    "acq_fn_type": trial.suggest_categorical(
                        "acq_fn_type", ["its", "ucb", "ei"]
                    ),
                    "encoding_type": None,
                    "num_candidates": trial.suggest_int("num_candidates", 50, 200),
                    "removal_percentage": 0.25,
                },
            },
            # Stage 2 configuration (ZCP GSparsity)
            "stage2": {
                "search": {
                    "epochs": search_epochs // 2,
                    "grad_clip": trial.suggest_categorical(
                        "grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                    ),
                    "weight_decay": trial.suggest_float(
                        "weight_decay", 30.0, 150.0, log=True
                    ),
                    "threshold": None,  # Not used in the current implementation of GSparsity
                    # "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                    "normalization": trial.suggest_categorical(
                        "normalization", ["none", "mul", "div"]
                    ),
                    "normalization_exponent": trial.suggest_float(
                        "normalization_exponent", 0.25, 0.75
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-4, 1e-1, log=True
                    ),
                    "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                    "learning_rate_min": trial.suggest_float(
                        "learning_rate_min", 1e-5, 5e-4, log=True
                    ),
                    "zcp_method": zcp_method,
                },
            },
        }
    elif optimizer_type == "rs":
        config = {
            "search": {
                "checkpoint_freq": 5,
                "epochs": search_epochs,
                "fidelity": -1,
            },
        }
    elif optimizer_type == "random_sampling":
        config = {
            "search": {
                "checkpoint_freq": 5,
                "epochs": 1,
                "fidelity": -1,
            },
        }
    else:
        # This will catch any optimizer types that are not configured for HPO
        raise ValueError(f"Optimizer '{optimizer_type}' not set up for HPO.")

    # Suggest cutout parameter for relevant optimizers
    if optimizer_type in [
        "gsparsity",
        "zcp_gsparsity",
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
        "zcp-pre_gsparsity",
        "zcp-pre_zcp_gsparsity",
        "self_training_inverted_bananas_gsparsity",
        "self_training_inverted_bananas_zcp_gsparsity",
    ]:
        cutout = trial.suggest_categorical("cutout", [False, True])
        config["search"]["cutout"] = cutout
        config["search"]["cutout_length"] = trial.suggest_int("cutout_length", 8, 24)
        config["search"]["cutout_prob"] = trial.suggest_float("cutout_prob", 0.1, 1.0)

        # cutout = trial.suggest_categorical("cutout", [False, True])
        # if cutout:
        #     config["search"]["cutout"] = True
        #     config["search"]["cutout_length"] = trial.suggest_int(
        #         "cutout_length", 8, 24
        #     )
        #     config["search"]["cutout_prob"] = trial.suggest_float(
        #         "cutout_prob", 0.1, 1.0
        #     )
        # else:
        #     config["search"]["cutout"] = False
        #     config["search"]["cutout_length"] = 0
        #     config["search"]["cutout_prob"] = None

    if optimizer_type in [
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
        "self_training_inverted_bananas_gsparsity",
        "self_training_inverted_bananas_zcp_gsparsity",
    ]:
        config["stage1"]["search"]["acq_fn_optimization"] = trial.suggest_categorical(
            "acq_fn_optimization", ["mutation", "random_sampling"]
        )
        config["stage1"]["search"]["num_arches_to_mutate"] = trial.suggest_int(
            "num_arches_to_mutate", 1, 5
        )
        config["stage1"]["search"]["max_mutations"] = trial.suggest_int(
            "max_mutations", 1, 3
        )
        # acq_fn_optimization = trial.suggest_categorical(
        #     "acq_fn_optimization", ["mutation", "random_sampling"]
        # )
        # if acq_fn_optimization == "mutation":
        #     config["stage1"]["search"]["acq_fn_optimization"] = "mutation"
        #     config["stage1"]["search"]["num_arches_to_mutate"] = trial.suggest_int(
        #         "num_arches_to_mutate", 1, 5
        #     )
        #     config["stage1"]["search"]["max_mutations"] = trial.suggest_int(
        #         "max_mutations", 1, 3
        #     )
        # else:
        #     config["stage1"]["search"]["acq_fn_optimization"] = "random_sampling"
        #     config["stage1"]["search"]["num_arches_to_mutate"] = 0
        #     config["stage1"]["search"]["max_mutations"] = 0

    # Add common evaluation config
    config["evaluation"] = evaluation

    config["search"]["early_stopping"] = {
        "criterion": "valid_loss",  # Can be 'train_acc', 'train_loss', 'valid_acc', 'valid_loss', or 'runtime'
        "patience": 10,  # Number of epochs to wait for improvement
        "threshold": 0.00001,  # Minimum change to be considered an improvement
    }

    # --- Start: New, Smarter Duplicate & Promotion Handling Logic ---
    pruner = trial.study.pruner
    current_budget = -1  # Default value if budget cannot be determined
    resume_from_trial_number = None  # Track which trial to resume from

    # Step 1: Calculate the intended budget for the current trial.
    if hasattr(pruner, "_get_bracket_id_after_init") and hasattr(
        pruner, "_get_budget_id"
    ):
        try:
            # Manually construct a FrozenTrial. `create_trial` does not accept `number`.
            # This object is a temporary representation used for budget calculation.
            frozen_trial_for_budget_calc = FrozenTrial(
                number=trial.number,
                state=TrialState.RUNNING,
                value=None,
                datetime_start=trial.datetime_start,
                datetime_complete=None,
                params=trial.params,
                distributions=trial.distributions,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=trial._trial_id,
            )
            if len(pruner._pruners) == 0:
                pruner._try_initialization(trial.study)
            if pruner._budget_candidates is None:
                pruner._create_budget_candidates(trial.study)

            bracket_id = pruner._get_bracket_id_after_init(
                trial.study, frozen_trial_for_budget_calc
            )
            budget_id = pruner._get_budget_id(
                trial.study, frozen_trial_for_budget_calc, bracket_id
            )
            current_budget = pruner._budget_candidates[budget_id]
            trial.set_user_attr("budget", current_budget)
            logging.info(
                f"Trial {trial.number} has an intended budget of {current_budget} steps."
            )
        except Exception as e:
            logging.warning(
                f"Could not determine budget for trial {trial.number}. Proceeding without reuse/resume logic. Error: {e}"
            )

    # Step 2: Find all previous trials with identical hyperparameters.
    current_params_hash = trial_hash(trial.params)
    all_trials = trial.study.get_trials(deepcopy=False)
    matching_trials = [
        t
        for t in all_trials
        if t.number != trial.number and trial_hash(t.params) == current_params_hash
    ]

    # Step 3: Apply the specified reuse/resume logic if matches are found and budget is known.
    if matching_trials and current_budget != -1:
        # Sort matching trials by their budget in descending order to easily find the highest.
        matching_trials.sort(key=lambda t: t.user_attrs.get("budget", 0), reverse=True)
        highest_budget_trial = matching_trials[0]
        highest_budget = highest_budget_trial.user_attrs.get("budget", 0)

        # Scenario 1: Exact Budget Match.
        # Find a trial that matches the current budget exactly.
        exact_match_trial = next(
            (
                t
                for t in matching_trials
                if t.user_attrs.get("budget") == current_budget
            ),
            None,
        )
        if exact_match_trial:
            # First, update config to get the destination path for copying
            temp_config = CfgNode.load_cfg(json.dumps({"search": {}, "evaluation": {}}))
            dest_trial_path = update_config(
                temp_config,
                optimizer_type,
                search_space_type,
                dataset,
                seed,
                out_dir,
                trial,
            ).save

            if exact_match_trial.state == TrialState.COMPLETE:
                logging.info(
                    f"Trial {trial.number} is an exact match for COMPLETED Trial {exact_match_trial.number} "
                    f"(budget {current_budget}). Reusing its value: {exact_match_trial.value}."
                )
                _copy_trial_artifacts(
                    exact_match_trial.number, dest_trial_path, out_dir
                )
                # Reporting the value at the specific budget step makes it a valid completed trial for the pruner.
                trial.report(exact_match_trial.value, current_budget)
                return exact_match_trial.value  # Reuse result and mark as COMPLETE

            elif exact_match_trial.state == TrialState.PRUNED:
                _copy_trial_artifacts(
                    exact_match_trial.number, dest_trial_path, out_dir
                )
                # The trial is a duplicate of a pruned trial. We should report the same
                # intermediate value that caused the original to be pruned, and then prune this one.
                last_step = exact_match_trial.last_step
                if last_step is not None:
                    last_value = exact_match_trial.intermediate_values[last_step]
                    trial.report(last_value, last_step)
                    logging.info(
                        f"Trial {trial.number} is an exact match for PRUNED Trial {exact_match_trial.number} "
                        f"(budget {current_budget}). Reporting its last value ({last_value} at step {last_step}) and pruning."
                    )
                else:
                    logging.info(
                        f"Trial {trial.number} is an exact match for PRUNED Trial {exact_match_trial.number} "
                        f"(budget {current_budget}), which had no reported values. Pruning immediately."
                    )
                raise optuna.TrialPruned()

        # Scenario 2: Promotion (Current budget is higher than any previous attempt).
        # This trial should resume from the checkpoint of the highest-budget previous trial.
        elif current_budget > highest_budget:
            if highest_budget_trial.user_attrs.get("internal_early_stopped", False):
                logging.info(
                    f"Skipping promotion for Trial {trial.number} because its candidate "
                    f"(Trial {highest_budget_trial.number}) was internally early-stopped. Returning early stopped value {highest_budget_trial.value}."
                )
                trial.user_attrs["internal_early_stopped"] = True
                trial.report(highest_budget_trial.value, current_budget)
                return highest_budget_trial.value  # Reuse result and mark as COMPLETE

            logging.info(
                f"Trial {trial.number} (budget {current_budget}) is a promotion from Trial "
                f"{highest_budget_trial.number} (budget {highest_budget}, state {highest_budget_trial.state})."
            )
            # Whether the previous was COMPLETE or PRUNED, we resume from its state to continue training.
            resume_from_trial_number = highest_budget_trial.number
            # Set a specific attribute for promotions. This is NOT a replacement.
            trial.set_user_attr("promoted_from", resume_from_trial_number)

        # If neither an exact match nor a promotion, it's a new trial for a specific rung.
        # The `else` block for redundancy has been removed as it was incorrect for Hyperband.

    # Step 4: Handle resuming the last interrupted (FAIL/RUNNING) trial if no other logic applied.
    # This is a fallback for crashes.
    if resume_from_trial_number is None:
        interrupted_trials = [
            t
            for t in matching_trials
            if t.state in [TrialState.FAIL, TrialState.RUNNING]
        ]
        if interrupted_trials:
            # Find the most recent interrupted trial to consider resuming.
            last_interrupted = max(interrupted_trials, key=lambda t: t.number)
            interrupted_budget = last_interrupted.user_attrs.get("budget", 0)

            # CRITICAL CHECK: Before resuming, ensure no completed/pruned trial with a higher budget exists.
            # If one does, the interrupted trial is obsolete and should not be resumed.
            has_more_advanced_trial = any(
                t.state in [TrialState.COMPLETE, TrialState.PRUNED]
                and t.user_attrs.get("budget", 0) > interrupted_budget
                for t in matching_trials
            )

            if not has_more_advanced_trial:
                logging.info(
                    f"Trial {trial.number} matches the last interrupted trial (Trial {last_interrupted.number}). "
                    "Setting up to resume as no more advanced trial exists."
                )
                resume_from_trial_number = last_interrupted.number
                # Set a specific attribute for resuming an interrupted trial. This IS a replacement.
                trial.set_user_attr(
                    "resumed_interrupted_from", resume_from_trial_number
                )
            else:
                logging.info(
                    f"Ignoring interrupted Trial {last_interrupted.number} because a more "
                    "advanced (higher budget) trial with the same hyperparameters already exists."
                )
    # --- End: New Logic ---

    # Convert dictionary to CfgNode
    config = CfgNode.load_cfg(json.dumps(config))

    # Set the dataset subset percentage for HPO #!
    config.dataset_subset = 0.001

    # Set epochs for HPO trial.
    # We hardcode this to 1 epoch for one-stage methods and 1+1 for two-stage methods.
    if current_budget != -1:
        config.search.epochs = current_budget
        logging.info(f"Setting trial epochs to the calculated budget: {current_budget}")
    else:
        # Fallback for one-stage methods if budget calculation fails
        # For two-stage methods, a different logic is applied below.
        if optimizer_type not in [
            "inverted_bananas_gsparsity",
            "inverted_bananas_zcp_gsparsity",
            "self_training_inverted_bananas_gsparsity",
            "self_training_inverted_bananas_zcp_gsparsity",
        ]:
            config.search.epochs = 9  # A default fallback
            logging.warning(
                f"Could not determine budget. Falling back to default epochs: {config.search.epochs}"
            )

    # We hardcode this to 1 epoch for one-stage methods and 1+1 for two-stage methods.
    if optimizer_type in [
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
        "self_training_inverted_bananas_gsparsity",
        "self_training_inverted_bananas_zcp_gsparsity",
    ]:
        # For two-stage methods, the budget is managed internally by the pruner.
        # We can set a max, but the pruner decides the actual epochs for each stage.
        # If budget was calculated, use it. Otherwise, fall back to a default.
        total_epochs = current_budget if current_budget != -1 else 70
        config.search.epochs = total_epochs
        # Split the budget between stages if needed, e.g., 40/60 split
        stage1_budget = int(config.search.epochs * 0.4)
        config.stage1.search.epochs = stage1_budget
        config.stage1.search.train_epochs = stage1_budget  # For self-training
        config.stage2.search.epochs = config.search.epochs - stage1_budget
        logging.info(
            f"Two-stage method epochs set. Total: {config.search.epochs}, Stage 1: {config.stage1.search.epochs}, Stage 2: {config.stage2.search.epochs}"
        )

    # Update config with other details
    config = update_config(
        config, optimizer_type, search_space_type, dataset, seed, out_dir, trial
    )

    # --- Handle resuming by copying files from the identified trial ---
    if resume_from_trial_number is not None:
        # Build source path with zcp_method if needed
        source_dir_parts = [
            out_dir,
            "WHPO",
            optimizer_type,
            search_space_type,
            dataset,
            str(seed),
        ]
        if "zcp_" in optimizer_type:
            source_dir_parts.append(zcp_method)
        source_dir_parts.append(f"trial_{resume_from_trial_number}")
        source_trial_path = os.path.join(*source_dir_parts)
        dest_trial_path = config.save

        if os.path.isdir(source_trial_path):
            logging.info(
                f"Copying entire resume directory from {source_trial_path} to {dest_trial_path}"
            )
            os.makedirs(dest_trial_path, exist_ok=True)
            shutil.copytree(source_trial_path, dest_trial_path, dirs_exist_ok=True)
        else:
            logging.warning(
                f"Source directory for resume not found: {source_trial_path}. Starting trial from scratch."
            )

    # Run the optimizer
    try:
        _, final_val_acc, _ = run_optimizer(
            optimizer_type, search_space_type, dataset, config, seed, trial
        )
        return final_val_acc
    except optuna.TrialPruned:
        # Let Optuna handle pruned trials
        raise
    except Exception as e:
        # Log the error and re-raise to let Optuna mark the trial as FAILED
        logging.error(f"Trial {trial.number} failed due to: {e}", exc_info=True)
        raise


def update_config(
    config, optimizer_type, search_space_type, dataset, seed, out_dir, trial
):
    """Update the configuration with experiment-specific settings"""
    # Set dataset name

    if (
        optimizer_type == "gsparsity"
        or optimizer_type == "zcp_gsparsity"
        or optimizer_type == "drnas"
        or optimizer_type == "inverted_bananas_gsparsity"
        or optimizer_type == "inverted_bananas_zcp_gsparsity"
        or optimizer_type == "self_training_inverted_bananas_gsparsity"
        or optimizer_type == "self_training_inverted_bananas_zcp_gsparsity"
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

    config.data = str(get_project_root()) + "/data"  # path to naslib/data directory
    print(f"Data path: {config.data}")

    # Set search space
    config.search_space = search_space_type

    # Set up output directory path
    # Using just the trial number is safer and cleaner than a long parameter string

    # The new trial will save its own logs and checkpoints to its unique directory.
    config.save = os.path.join(
        out_dir,
        "WHPO",
        optimizer_type,
        search_space_type,
        dataset,
        str(seed),
    )
    # Add zcp_method to path for ZCP optimizers
    if "zcp_" in optimizer_type:
        config.save = os.path.join(config.save, zcp_method)
    config.save = os.path.join(config.save, f"trial_{trial.number}")

    # This logic is now handled by copying files in the objective function.
    # config.resume_from_path is no longer needed.

    if "inverted_bananas_zcp_gsparsity" in optimizer_type:
        config.stage2.search.zcp_method = zcp_method
    elif "zcp_gsparsity" in optimizer_type:
        config.search.zcp_method = zcp_method

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


def run_optimizer(optimizer_type, search_space_type, dataset, config, seed, trial):
    """Run the optimizer with the specified configuration"""
    # Clear handlers from previous trials to prevent duplicate logging and deadlocks
    _clear_logging_handlers()

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
        # The resume logic now simply checks the current trial's directory.
        # The objective function is responsible for populating it with files from a failed run.
        search_resume_from = utils.get_last_checkpoint(config, search=True)
        if search_resume_from:
            logging.info(
                f"Resume is True. Found checkpoint in current trial directory: {search_resume_from}"
            )
        else:
            logging.info(
                f"Resume is True, but no checkpoint found in '{config.save}/search'. Starting from scratch."
            )
    else:
        logging.info("Resume is False. Starting from scratch.")

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
    logging.info("Loading Benchmark API")
    dataset_api = get_dataset_api(search_space_type, dataset)

    # Instantiate the optimizer
    if optimizer_type in ["rs", "random_sampling"]:
        optimizer = RandomSearch(config)
    elif optimizer_type == "gsparsity":
        optimizer = GSparseOptimizer(config)
    elif optimizer_type == "zcp_gsparsity":
        optimizer = ZCP_GSparseOptimizer(config)
    elif optimizer_type == "inverted_bananas_gsparsity":
        optimizer = Inverted_Bananas_GsparseOptimizer(config)
    elif optimizer_type == "inverted_bananas_zcp_gsparsity":
        optimizer = Inverted_Bananas_ZCP_GsparseOptimizer(config)
    elif optimizer_type == "self_training_inverted_bananas_gsparsity":
        optimizer = SelfTrainingInvertedBananasGsparse(config)
    elif optimizer_type == "self_training_inverted_bananas_zcp_gsparsity":
        optimizer = SelfTrainingInvertedBananasZCPGsparse(config)
    elif optimizer_type == "zcp-pre_gsparsity":
        optimizer = PreZCPGSParseOptimizer(config)
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
        if "zcp_" in optimizer_type:
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
        from naslib.defaults.two_stage_trainer_multi_dataloading_workers import Trainer

        trainer = Trainer(optimizer, config, lightweight_output=False)
    else:
        from naslib.defaults.trainer_multi_dataloading_workers import Trainer

        trainer = Trainer(
            optimizer, config, lightweight_output=False
        )  #! just changed to true

    trainer.search(
        resume_from=search_resume_from,
        report_incumbent=False,
        trial=trial,
    )

    # Get the search trajectory
    search_trajectory = trainer.search_trajectory

    # The objective function needs the final validation accuracy and loss from the search.
    final_val_acc = -1.0
    final_val_loss = float("inf")  # Initialize with a high value for minimization

    if search_trajectory and search_trajectory.valid_acc:
        final_val_acc = search_trajectory.valid_acc[-1]

    if (
        search_trajectory
        and search_trajectory.valid_loss
        and search_trajectory.valid_loss[-1] is not None
    ):
        final_val_loss = search_trajectory.valid_loss[-1]

    # During HPO, we don't need to run the full evaluation.
    logger.info(
        f"Finished trial search. Final validation accuracy: {final_val_acc}, Final validation loss: {final_val_loss}"
    )

    return search_trajectory, final_val_acc, final_val_loss


def main():
    """Main function to run the HPO study"""
    # Set the start method for multiprocessing to 'spawn' to avoid deadlocks with CUDA.
    # This needs to be done in the main process before any subprocesses are created.
    # try:
    #     if mp.get_start_method(allow_none=True) != "spawn":
    #         mp.set_start_method("spawn", force=True)
    #         logging.info("Set multiprocessing start method to 'spawn'.")
    # except RuntimeError:
    #     # This can be raised if the context is already started.
    #     logging.warning("Could not set multiprocessing start method.")
    #     pass

    # DEHB setup
    module = optunahub.load_module("samplers/dehb", force_reload=False)
    DEHBSampler = module.DEHBSampler
    DEHBPruner = module.DEHBPruner

    # Determine max_resource for the pruner
    if optimizer_type in [
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
        "self_training_inverted_bananas_gsparsity",
        "self_training_inverted_bananas_zcp_gsparsity",
    ]:
        max_res = 70
        min_res = 21
        # min_res = 1
    else:
        max_res = 9
        min_res = 1

    sampler = DEHBSampler(seed=seed)
    pruner = DEHBPruner(min_resource=min_res, max_resource=max_res, reduction_factor=3)

    # Define storage path for the Optuna study database
    # This creates a unique database for each optimizer/dataset combination
    # inside the main output directory.
    db_dir = os.path.join(out_dir, "WHPO_Databases")
    os.makedirs(db_dir, exist_ok=True)

    # Construct a unique study name and database file path
    study_name_parts = [optimizer_type, search_space_type, dataset, str(seed)]
    if "zcp_" in optimizer_type:
        study_name_parts.append(zcp_method)

    study_name = "-".join(study_name_parts)
    db_filename = f"{study_name}.db"
    storage_name = f"sqlite:///{os.path.join(db_dir, db_filename)}"

    print(f"Using Optuna study '{study_name}' in database '{storage_name}'")

    # Create study
    study = optuna.create_study(
        storage=storage_name,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",  # We maximize validation accuracy
        study_name=study_name,
    )

    # Start optimization
    try:
        study.optimize(objective, timeout=hpo_timeout)
    except Exception as e:
        print(f"An exception occurred during the study: {e}")

    # # --- Hyperparameter Importance Analysis (Post-Run) ---
    # print("\n--- Hyperparameter Importance Analysis ---")
    # try:
    #     # By default, get_param_importances only uses successfully completed trials (TrialState.COMPLETE).
    #     # This is the correct behavior, as pruned trials do not have a final, comparable objective value.
    #     completed_trials = study.get_trials(
    #         deepcopy=False, states=[TrialState.COMPLETE]
    #     )
    #     if len(completed_trials) > 1:
    #         param_importances = optuna.importance.get_param_importances(study)

    #         print("Parameter importances (fANOVA):")
    #         sorted_importances = sorted(
    #             param_importances.items(), key=lambda x: x[1], reverse=True
    #         )
    #         for param, importance in sorted_importances:
    #             print(f"  {param}: {importance:.4f}")

    #         # Visualize and save the importances plot
    #         fig = optuna.visualization.plot_param_importances(study)
    #         plot_path = os.path.join(
    #             out_dir, f"{optimizer_type}_{dataset}_{seed}_param_importances.html"
    #         )
    #         fig.write_html(plot_path)
    #         print(f"\nSaved parameter importance plot to: {plot_path}")
    #     else:
    #         print(
    #             "Skipping importance analysis: not enough completed trials to analyze."
    #         )

    # except Exception as e:
    #     print(f"Could not calculate or plot parameter importances: {e}")
    # # --- End of Analysis ---

    # # Print results
    # pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    # complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # print("\nStudy statistics: ")
    # print(f"  Number of finished trials: {len(study.trials)}")
    # print(f"  Number of pruned trials: {len(pruned_trials)}")
    # print(f"  Number of complete trials: {len(complete_trials)}")

    # print("Best trial:")
    # trial = study.best_trial

    # print(f"  Value: {trial.value}")

    # print("  Params: ")
    # for key, value in trial.params.items():
    #     print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
