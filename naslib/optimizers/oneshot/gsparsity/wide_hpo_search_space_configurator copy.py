import logging
import os
from fvcore.common.config import CfgNode
import json
import argparse

import optuna
import optunahub
import copy
import time

from optuna.trial import TrialState

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
    help="Optimizer type (rs, ls, bananas, drnas, gsparsity, zcp_gsparsity, inverted_bananas, inverted_bananas_gsparsity, inverted_bananas_zcp_gsparsity)",
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


# Maybe the problem is also related to the change of the logger in the configurator. As the statedict in the log says it is not complete thus the checkpoint after completing one epoch with gsparsity should also contain less. Or is this the case and my logging / printing information is just not nuanced enough to capture this?

# have look at log.log file


#! update logging to capture the hp settings

#! check standard value ranges for each hyperparameter (papers for resnet and conv nets)

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


def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for Optuna HPO.
    """
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
                "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                "normalization": trial.suggest_categorical(
                    "normalization", ["none", "mul", "div"]
                ),
                "normalization_exponent": trial.suggest_float(
                    "normalization_exponent", 0.25, 0.75
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True
                ),
                "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                "learning_rate_min": trial.suggest_float(
                    "learning_rate_min", 1e-5, 5e-4, log=True
                ),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "train_portion": trial.suggest_float("train_portion", 0.9, 1.0),
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
                "threshold": trial.suggest_float("threshold", 1e-7, 1e-4, log=True),
                "normalization": trial.suggest_categorical(
                    "normalization", ["none", "mul", "div"]
                ),
                "normalization_exponent": trial.suggest_float(
                    "normalization_exponent", 0.25, 0.75
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True
                ),
                "momentum": trial.suggest_float("momentum", 0.7, 0.95),
                "learning_rate_min": trial.suggest_float(
                    "learning_rate_min", 1e-5, 5e-4, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "zcp_gsparsitybatch_size", [32, 64, 128]
                ),
                "train_portion": trial.suggest_float("train_portion", 0.9, 1.0),
                "zcp_method": zcp_method,
            },
        }
    elif optimizer_type == "inverted_bananas_gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "batch_size": trial.suggest_categorical(
                    "ibg_batch_size", [32, 64, 128]
                ),
                "train_portion": trial.suggest_float("ibg_train_portion", 0.4, 0.6),
            },
            # Stage 1 configuration (Inverted BANANAS)
            "stage1": {
                "search": {
                    "epochs": search_epochs // 2,
                    "k": trial.suggest_int("ibg_s1_k", 5, 20),
                    "num_init": trial.suggest_int("ibg_s1_num_init", 5, 20),
                    "num_ensemble": 5,
                    "predictor_type": trial.suggest_categorical(
                        "ibg_s1_predictor_type", ["mlp", "lgb", "xgb", "rf"]
                    ),
                    "acq_fn_type": trial.suggest_categorical(
                        "ibg_s1_acq_fn_type", ["its", "ucb", "ei"]
                    ),
                    "acq_fn_optimization": "mutation",
                    "encoding_type": None,
                    "num_arches_to_mutate": 1,
                    "max_mutations": 1,
                    "num_candidates": 100,
                    "removal_percentage": trial.suggest_float(
                        "ibg_s1_removal_percentage", 0.5, 1.0
                    ),
                },
            },
            # Stage 2 configuration (GSparsity)
            "stage2": {
                "search": {
                    "epochs": search_epochs // 2,
                    "grad_clip": trial.suggest_categorical(
                        "ibg_s2_grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                    ),
                    "weight_decay": trial.suggest_float(
                        "ibg_s2_weight_decay", 30.0, 150.0, log=True
                    ),
                    "threshold": trial.suggest_float(
                        "ibg_s2_threshold", 1e-7, 1e-4, log=True
                    ),
                    "normalization": trial.suggest_categorical(
                        "ibg_s2_normalization", ["none", "mul", "div"]
                    ),
                    "normalization_exponent": trial.suggest_float(
                        "ibg_s2_normalization_exponent", 0.25, 0.75
                    ),
                    "learning_rate": trial.suggest_float(
                        "ibg_s2_learning_rate", 1e-4, 1e-2, log=True
                    ),
                    "momentum": trial.suggest_float("ibg_s2_momentum", 0.7, 0.95),
                    "learning_rate_min": trial.suggest_float(
                        "ibg_s2_learning_rate_min", 1e-5, 5e-4, log=True
                    ),
                },
            },
        }
    elif optimizer_type == "inverted_bananas_zcp_gsparsity":
        config = {
            "search": {
                "checkpoint_freq": 1,
                "epochs": search_epochs,
                "batch_size": trial.suggest_categorical(
                    "ibzg_batch_size", [32, 64, 128]
                ),
                "train_portion": trial.suggest_float("ibzg_train_portion", 0.4, 0.6),
            },
            # Stage 1 configuration (Inverted BANANAS)
            "stage1": {
                "search": {
                    "epochs": search_epochs // 2,
                    "k": trial.suggest_int("ibzg_s1_k", 5, 20),
                    "num_init": trial.suggest_int("ibzg_s1_num_init", 5, 20),
                    "num_ensemble": 5,
                    "predictor_type": trial.suggest_categorical(
                        "ibzg_s1_predictor_type", ["mlp", "lgb", "xgb", "rf"]
                    ),
                    "acq_fn_type": trial.suggest_categorical(
                        "ibzg_s1_acq_fn_type", ["its", "ucb", "ei"]
                    ),
                    "acq_fn_optimization": "mutation",
                    "encoding_type": "path",
                    "num_arches_to_mutate": 1,
                    "max_mutations": 1,
                    "num_candidates": 100,
                    "removal_percentage": trial.suggest_float(
                        "ibzg_s1_removal_percentage", 0.5, 1.0
                    ),
                },
            },
            # Stage 2 configuration (ZCP GSparsity)
            "stage2": {
                "search": {
                    "epochs": search_epochs // 2,
                    "grad_clip": trial.suggest_categorical(
                        "ibzg_s2_grad_clip", [None, 0.5, 1.0, 5.0, 10.0]
                    ),
                    "weight_decay": trial.suggest_float(
                        "ibzg_s2_weight_decay", 30.0, 150.0, log=True
                    ),
                    "threshold": trial.suggest_float(
                        "ibzg_s2_threshold", 1e-7, 1e-4, log=True
                    ),
                    "normalization": trial.suggest_categorical(
                        "ibzg_s2_normalization", ["none", "mul", "div"]
                    ),
                    "normalization_exponent": trial.suggest_float(
                        "ibzg_s2_normalization_exponent", 0.25, 0.75
                    ),
                    "learning_rate": trial.suggest_float(
                        "ibzg_s2_learning_rate", 1e-4, 1e-2, log=True
                    ),
                    "momentum": trial.suggest_float("ibzg_s2_momentum", 0.7, 0.95),
                    "learning_rate_min": trial.suggest_float(
                        "ibzg_s2_learning_rate_min", 1e-5, 5e-4, log=True
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
    else:
        # This will catch any optimizer types that are not configured for HPO
        raise ValueError(f"Optimizer '{optimizer_type}' not set up for HPO.")

    # Suggest cutout parameter for relevant optimizers
    if optimizer_type in [
        "gsparsity",
        "zcp_gsparsity",
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
    ]:
        cutout = trial.suggest_categorical("cutout", [False, True])
        if cutout:
            config["search"]["cutout"] = True
            config["search"]["cutout_length"] = trial.suggest_int(
                "cutout_length", 8, 24
            )
            config["search"]["cutout_prob"] = trial.suggest_float(
                "cutout_prob", 0.1, 0.7
            )
        else:
            config["search"]["cutout"] = False
            config["search"]["cutout_length"] = 0
            config["search"]["cutout_prob"] = None

    # Add common evaluation config
    config["evaluation"] = evaluation

    # Convert dictionary to CfgNode
    config = CfgNode.load_cfg(json.dumps(config))

    # Set epochs for HPO trial
    if optimizer_type in [
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
    ]:
        config.search.epochs = 2
        config.stage1.search.epochs = 1
        config.stage2.search.epochs = 1
    else:
        config.search.epochs = 1

    # Update config with other details
    config = update_config(
        config, optimizer_type, search_space_type, dataset, seed, out_dir, trial
    )

    # Run the optimizer
    try:
        _, best_val_acc = run_optimizer(
            optimizer_type, search_space_type, dataset, config, seed, trial
        )
        return best_val_acc
    except optuna.TrialPruned:
        return -1.0  # Return a low value for pruned trials


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
    ):
        config.save_arch_weights = False

    config.dataset = dataset

    config.data = str(get_project_root()) + "/data"  # path to naslib/data directory
    print(f"Data path: {config.data}")

    # Set search space
    config.search_space = search_space_type

    # Set up output directory path
    # Using just the trial number is safer and cleaner than a long parameter string
    params_str = f"trial_{trial.number}"

    if "inverted_bananas_zcp_gsparsity" in optimizer_type:
        config.save = f"{out_dir}/{optimizer_type}/{config.stage2.search.zcp_method}/{search_space_type}/{dataset}/{seed}/{params_str}"
    elif "zcp_gsparsity" in optimizer_type:
        config.save = f"{out_dir}/{optimizer_type}/{config.search.zcp_method}/{search_space_type}/{dataset}/{seed}/{params_str}"
    else:
        config.save = f"{out_dir}/{optimizer_type}/{search_space_type}/{dataset}/{seed}/{params_str}"

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
    elif optimizer_type == "gsparsity":
        optimizer = GSparseOptimizer(config)
    elif optimizer_type == "zcp_gsparsity":
        optimizer = ZCP_GSparseOptimizer(config)
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
        optimizer.adapt_search_space(
            search_space=search_space, train_loader=train_loader
        )
    elif optimizer_type in ["rs", "ls", "bananas", "inverted_bananas"]:
        optimizer.adapt_search_space(search_space=search_space, dataset_api=dataset_api)
    elif optimizer_type in [
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
    ]:
        # For two-stage optimizers
        train_loader = None
        if "zcp" in optimizer_type:
            train_loader, _, _, _, _ = get_train_val_loaders(config)
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
    ]:
        from naslib.defaults.two_stage_trainer_multi_dataloading_workers import Trainer

        trainer = Trainer(optimizer, config, lightweight_output=False)
    else:
        from naslib.defaults.trainer_multi_dataloading_workers import Trainer

        trainer = Trainer(optimizer, config, lightweight_output=False)

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
    trainer.search(resume_from=search_resume_from, report_incumbent=False, trial=trial)

    # Get the search trajectory
    search_trajectory = trainer.search_trajectory

    # The objective function needs the final validation accuracy from the search.
    final_val_acc = -1.0
    if search_trajectory and search_trajectory.valid_acc:
        final_val_acc = search_trajectory.valid_acc[-1]

    # During HPO, we don't need to run the full evaluation.
    logger.info(f"Finished trial search. Final validation accuracy: {final_val_acc}")

    return search_trajectory, final_val_acc


def main():
    """Main function to run the HPO study"""
    # DEHB setup
    module = optunahub.load_module("samplers/dehb")
    DEHBSampler = module.DEHBSampler
    DEHBPruner = module.DEHBPruner

    # Determine max_resource for the pruner
    if optimizer_type in [
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
    ]:
        max_res = 2
    else:
        max_res = 1

    sampler = DEHBSampler(seed=seed)
    pruner = DEHBPruner(min_resource=1, max_resource=max_res, reduction_factor=3)

    # Create study
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        study_name=f"{optimizer_type}-{search_space_type}-{dataset}-{seed}",
    )

    # Start optimization
    try:
        study.optimize(objective, timeout=900)
    except Exception as e:
        print(f"An exception occurred during the study: {e}")

    # Print results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
