import argparse
import os
import json
import time
import logging
import numpy as np
import torch
from fvcore.common.config import CfgNode

from naslib.search_spaces import NasBench201SearchSpace
from naslib import utils
from naslib.utils import (
    setup_logger,
    get_dataset_api,
    get_project_root,
)
from naslib.utils.dataset import get_train_val_loaders
from naslib.search_spaces.core.query_metrics import Metric
from naslib.defaults.trainer import Trainer

# For Nesterov
from torch.optim.sgd import SGD

# Add for manual splitting
from torch.utils.data import Subset, DataLoader

# Basic configuration template
config_template = {
    "dataset": "cifar10",
    "data": str(get_project_root() / "data"),
    "search_space": "nasbench201",
    "optimizer": "eval_only",  # Placeholder, not used for NAS optimizer
    "seed": 42,
    "out_dir": "comp_factor_exp",  # Placeholder, will be set by args
    "search": {  # Minimal search config, not really used for search phase
        "seed": 42,
        "epochs": 1,  # Placeholder
        # For get_train_val_loaders if it looks here for batch_size/train_portion
        "batch_size": 256,
        "train_portion": 0.5,
    },
    "evaluation": {
        "epochs": 200,
        "batch_size": 256,
        "learning_rate": 0.1,
        "learning_rate_min": 0.0,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "grad_clip": 5,
        "drop_path_prob": 0.0,  # NB201 paper doesn't mention drop_path for training
        "cutout": False,  # NB201 paper doesn't mention cutout
        "cutout_length": 16,  # Default if cutout were true
        "cutout_prob": 1.0,  # Default if cutout were true
        "train_portion": 0.5,  # Will be overridden based on dataset
        # Nesterov needs to be handled in optimizer creation
    },
}
config = CfgNode.load_cfg(json.dumps(config_template))


def main(args):
    # Setup output dir and logger
    log_dir = os.path.join(args.out_dir, args.dataset, str(args.seed))
    os.makedirs(log_dir, exist_ok=True)

    results_dir = os.path.join(args.out_dir, args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    results_json_path = os.path.join(results_dir, "results.json")

    logger = setup_logger(os.path.join(log_dir, "log.log"))
    logger.setLevel(logging.INFO)
    logger.info(f"Script arguments: {args}")
    logger.info(f"Logging to: {os.path.join(log_dir, 'log.log')}")
    logger.info(f"Results will be saved to/appended to: {results_json_path}")

    # Create config from template and args
    cfg = config.clone()  # Clone the global CfgNode
    cfg.dataset = args.dataset
    cfg.seed = args.seed
    cfg.evaluation.epochs = args.epochs
    cfg.evaluation.batch_size = args.batch_size
    # Ensure search section also has batch_size for get_train_val_loaders
    # train_portion will be set below per dataset for both evaluation and search contexts
    cfg.search.batch_size = args.batch_size

    if args.dataset == "cifar10":
        # Paper: 50K original train -> 25K new train, 25K new val. Test set is original 10K.
        # NASLib's get_train_val_loaders splits the 50K original train.
        paper_cifar10_original_train_size = 50000
        paper_cifar10_val_set_size = 25000
        cfg.evaluation.train_portion = (
            paper_cifar10_original_train_size - paper_cifar10_val_set_size
        ) / paper_cifar10_original_train_size  # This is 0.5
        cfg.search.train_portion = cfg.evaluation.train_portion
        n_classes = 10
        logger.info(
            f"CIFAR-10: Paper splits 50k original train into 25k train / 25k val. Original test set is 10k. "
            f"NASLib's `get_train_val_loaders` with train_portion={cfg.evaluation.train_portion:.4f} will be used. "
            f"Expected NASLib internal train queue size (Paper's Train): {int(cfg.evaluation.train_portion * paper_cifar10_original_train_size)}, "
            f"Expected NASLib internal val queue size (Paper's Val): {paper_cifar10_val_set_size}. "
            f"NASLib internal test queue (Paper's Test) will use the original 10k CIFAR-10 test set."
        )

    elif args.dataset == "cifar100":
        # Paper: Train 50k (original 50k train). Val 5k (from original 10k test). Test 5k (from original 10k test).
        # NASLib's get_train_val_loaders splits the 50K original train for its train/val queues.
        # We want NASLib's train_queue to be the full 50k.
        cfg.evaluation.train_portion = (
            1.0  # Use all original training data for NASLib's train_queue
        )
        cfg.search.train_portion = cfg.evaluation.train_portion
        n_classes = 100
        logger.info(
            f"CIFAR-100: Paper uses 50k for training, 5k from original test for validation, 5k from original test for test. "
            f"Setting train_portion={cfg.evaluation.train_portion:.4f} for NASLib's `get_train_val_loaders`. "
            f"Expected NASLib internal train queue size (Paper's Train): 50000. "
            f"NASLib internal val queue from `get_train_val_loaders` will be empty/tiny and ignored. "
            f"The 10k original CIFAR-100 test set (from NASLib's test_queue) will be split into 5k for Paper's Val and 5k for Paper's Test."
        )

    elif args.dataset == "ImageNet16-120":
        # Paper: Train 151.7K. Val 3K (from original 6k test). Test 3K (from original 6k test).
        # NASLib's ImageNet16 provides ~151.7K 'train' data and 6K 'test' data.
        # We want NASLib's train_queue to be the full ~151.7k.
        cfg.evaluation.train_portion = (
            1.0  # Use all original training data for NASLib's train_queue
        )
        cfg.search.train_portion = cfg.evaluation.train_portion
        n_classes = 120
        logger.info(
            f"ImageNet16-120: Paper uses ~151.7k for training, 3k from original test for validation, 3k from original test for testing. "
            f"Setting train_portion={cfg.evaluation.train_portion:.4f} for NASLib's `get_train_val_loaders`. "
            f"Expected NASLib internal train queue size (Paper's Train): ~151700. "
            f"NASLib internal val queue from `get_train_val_loaders` will be empty/tiny and ignored. "
            f"The 6k original ImageNet16-120 test set (from NASLib's test_queue) will be split into 3k for Paper's Val and 3k for Paper's Test."
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    utils.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_api = get_dataset_api(cfg.search_space, cfg.dataset)

    arch_op_indices_list = []
    if args.arch_indices:
        try:
            arch_op_indices_list = json.loads(args.arch_indices)
            if not isinstance(arch_op_indices_list, list) or not all(
                isinstance(sublist, list) for sublist in arch_op_indices_list
            ):
                raise ValueError
            logger.info(
                f"Using provided list of {len(arch_op_indices_list)} architectures."
            )
        except (json.JSONDecodeError, ValueError):
            logger.error(
                f"Invalid format for --arch_indices. Expected a JSON string of a list of lists. Got: {args.arch_indices}"
            )
            logger.info(f"Sampling {args.num_archs} random architectures instead.")
            arch_op_indices_list = []  # Fallback to random sampling

    if not arch_op_indices_list:  # If not provided or parsing failed
        for _ in range(args.num_archs):
            # Ensure we get a *new* random arch each time by re-sampling
            current_arch_space = NasBench201SearchSpace(n_classes=n_classes)
            current_arch_space.sample_random_architecture(dataset_api=dataset_api)
            arch_op_indices_list.append(current_arch_space.get_op_indices())

    unique_arch_op_indices_list = []
    for item in arch_op_indices_list:  # Ensure uniqueness if sampling led to duplicates
        if item not in unique_arch_op_indices_list:
            unique_arch_op_indices_list.append(item)
    arch_op_indices_list = unique_arch_op_indices_list
    logger.info(
        f"Processing {len(arch_op_indices_list)} unique architectures: {arch_op_indices_list}"
    )

    # Load existing results if the file exists
    results = []
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, "r") as f:
                existing_results = json.load(f)
            if isinstance(existing_results, list):
                results = existing_results
                logger.info(
                    f"Loaded {len(results)} existing results from {results_json_path}"
                )
            else:
                logger.warning(
                    f"Existing results file {results_json_path} does not contain a list. Starting fresh."
                )
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(
                f"Could not load or parse existing results from {results_json_path}: {e}. Starting fresh."
            )

    for i, op_indices in enumerate(arch_op_indices_list):
        logger.info(
            f"Processing architecture {i + 1}/{len(arch_op_indices_list)}: {op_indices}"
        )

        # 1. Query from NAS-Bench-201
        queried_arch = NasBench201SearchSpace(n_classes=n_classes)
        queried_arch.set_op_indices(op_indices)

        query_epoch = 199  # NAS-Bench-201 was trained for 200 epochs (0-199)
        queried_train_time = queried_arch.query(
            Metric.TRAIN_TIME, cfg.dataset, dataset_api=dataset_api
        )
        queried_train_acc = queried_arch.query(
            Metric.TRAIN_ACCURACY,
            cfg.dataset,
            dataset_api=dataset_api,
            epoch=query_epoch,
        )
        queried_val_acc = queried_arch.query(
            Metric.VAL_ACCURACY,
            cfg.dataset,
            dataset_api=dataset_api,
            epoch=query_epoch,
        )
        queried_test_acc = queried_arch.query(
            Metric.TEST_ACCURACY,
            cfg.dataset,
            dataset_api=dataset_api,
            epoch=query_epoch,
        )
        logger.info(
            f"  Queried: TrainTime={queried_train_time:.2f}s, TrainAcc={queried_train_acc:.4f}, ValAcc={queried_val_acc:.4f}, TestAcc={queried_test_acc:.4f}"
        )

        # 2. Local Training
        arch_to_train = NasBench201SearchSpace(n_classes=n_classes)
        arch_to_train.set_op_indices(op_indices)
        arch_to_train.parse()
        arch_to_train.to(device)

        train_queue_naslib, valid_queue_naslib, test_queue_naslib, _, _ = (
            get_train_val_loaders(
                cfg,
                mode="val",
            )
        )

        # Log initial sizes from get_train_val_loaders
        logger.info(
            f"NASLib internal train queue size (from sampler): {len(train_queue_naslib.sampler.indices) if hasattr(train_queue_naslib.sampler, 'indices') else len(train_queue_naslib.dataset)}"
        )
        if (
            hasattr(valid_queue_naslib.sampler, "indices")
            and len(valid_queue_naslib.sampler.indices) > 0
        ):
            logger.info(
                f"NASLib internal val queue size (from sampler): {len(valid_queue_naslib.sampler.indices)}"
            )
        else:
            logger.info(
                "NASLib internal val queue (from sampler) is empty or not using a subset sampler."
            )

        logger.info(
            f"NASLib internal test queue size (full original test set): {len(test_queue_naslib.dataset)}"
        )

        # Assign queues for training based on dataset strategy
        train_queue_final = train_queue_naslib

        if args.dataset == "cifar10":
            # For CIFAR-10, NASLib's splits match the paper's.
            valid_queue_final = valid_queue_naslib
            test_queue_final = test_queue_naslib
            logger.info(
                "Using NASLib internal queues for CIFAR-10 as they match paper's splits."
            )
            logger.info(
                f"  Final Train queue size: {len(train_queue_final.sampler.indices) if hasattr(train_queue_final.sampler, 'indices') else len(train_queue_final.dataset)}"
            )
            logger.info(
                f"  Final Val queue size: {len(valid_queue_final.sampler.indices) if hasattr(valid_queue_final.sampler, 'indices') else len(valid_queue_final.dataset)}"
            )
            logger.info(f"  Final Test queue size: {len(test_queue_final.dataset)}")

        elif args.dataset == "cifar100":
            # Paper: Train 50k, Val 5k (from test), Test 5k (from test). Original test is 10k.
            original_test_set = test_queue_naslib.dataset
            num_original_test = len(original_test_set)  # Should be 10000

            paper_val_size = 5000
            paper_test_size = 5000

            assert num_original_test == paper_val_size + paper_test_size, (
                "CIFAR-100 original test set size mismatch for paper split"
            )

            # Create a permutation of indices for splitting
            # Ensure this split is deterministic for a given seed (np.random is seeded by utils.set_seed)
            indices = np.random.permutation(num_original_test)
            val_indices = indices[:paper_val_size]
            test_indices = indices[paper_val_size:]

            val_subset = Subset(original_test_set, val_indices)
            test_subset = Subset(original_test_set, test_indices)

            valid_queue_final = DataLoader(
                val_subset,
                batch_size=cfg.evaluation.batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )
            test_queue_final = DataLoader(
                test_subset,
                batch_size=cfg.evaluation.batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )

            logger.info(
                "CIFAR-100: Manually splitting NASLib internal test queue for Paper's Val and Test sets."
            )
            logger.info(
                f"  Final Train queue size: {len(train_queue_final.sampler.indices) if hasattr(train_queue_final.sampler, 'indices') else len(train_queue_final.dataset)}"
            )
            logger.info(
                f"  Final Val queue size (Paper's Val from original test): {len(valid_queue_final.dataset)}"
            )
            logger.info(
                f"  Final Test queue size (Paper's Test from original test): {len(test_queue_final.dataset)}"
            )

        elif args.dataset == "ImageNet16-120":
            # Paper: Train ~151.7k, Val 3k (from test), Test 3k (from test). Original test is 6k.
            original_test_set = test_queue_naslib.dataset
            num_original_test = len(original_test_set)  # Should be 6000

            paper_val_size = 3000
            paper_test_size = 3000

            assert num_original_test == paper_val_size + paper_test_size, (
                "ImageNet16-120 original test set size mismatch for paper split"
            )

            indices = np.random.permutation(num_original_test)
            val_indices = indices[:paper_val_size]
            test_indices = indices[paper_val_size:]

            val_subset = Subset(original_test_set, val_indices)
            test_subset = Subset(original_test_set, test_indices)

            valid_queue_final = DataLoader(
                val_subset,
                batch_size=cfg.evaluation.batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )
            test_queue_final = DataLoader(
                test_subset,
                batch_size=cfg.evaluation.batch_size,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )

            logger.info(
                "ImageNet16-120: Manually splitting NASLib internal test queue for Paper's Val and Test sets."
            )
            logger.info(
                f"  Final Train queue size: {len(train_queue_final.sampler.indices) if hasattr(train_queue_final.sampler, 'indices') else len(train_queue_final.dataset)}"
            )
            logger.info(
                f"  Final Val queue size (Paper's Val from original test): {len(valid_queue_final.dataset)}"
            )
            logger.info(
                f"  Final Test queue size (Paper's Test from original test): {len(test_queue_final.dataset)}"
            )
        else:
            # Fallback or error for unsupported datasets for this specific splitting logic
            logger.error(
                f"Dataset {args.dataset} does not have a specific paper split strategy implemented here. Using NASLib default splits."
            )
            valid_queue_final = valid_queue_naslib
            test_queue_final = test_queue_naslib

        optimizer = SGD(
            arch_to_train.parameters(),
            lr=cfg.evaluation.learning_rate,
            momentum=cfg.evaluation.momentum,
            weight_decay=cfg.evaluation.weight_decay,
            nesterov=True,
        )
        # Pass the full cfg to build_eval_scheduler
        scheduler = Trainer.build_eval_scheduler(optimizer, cfg)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        local_train_time_total = 0
        best_val_acc_local = 0
        final_train_acc_local = 0  # Train accuracy of the final epoch

        for epoch in range(cfg.evaluation.epochs):
            epoch_start_time = time.time()
            arch_to_train.train()
            train_loss_meter = utils.AverageMeter()
            train_acc_meter = utils.AverageMeter()

            for step, (input_train, target_train) in enumerate(
                train_queue_final
            ):  # Use final queue
                input_train = input_train.to(device)
                target_train = target_train.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits_train = arch_to_train(input_train)
                loss = criterion(logits_train, target_train)
                loss.backward()
                if cfg.evaluation.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        arch_to_train.parameters(), cfg.evaluation.grad_clip
                    )
                optimizer.step()

                prec1, _ = utils.accuracy(logits_train, target_train, topk=(1, 5))
                train_loss_meter.update(loss.item(), input_train.size(0))
                train_acc_meter.update(prec1.item(), input_train.size(0))

                if step % args.log_freq == 0 and step > 0:
                    logger.debug(
                        f"  Arch {i + 1} Epoch {epoch}/{cfg.evaluation.epochs - 1}, Step {step}/{len(train_queue_final) - 1}, TrainLoss: {train_loss_meter.avg:.4f}, TrainAcc: {train_acc_meter.avg:.4f}"  # Use final queue
                    )

            final_train_acc_local = train_acc_meter.avg

            arch_to_train.eval()
            val_acc_meter = utils.AverageMeter()
            if (
                valid_queue_final
            ):  # Ensure valid_queue_final is not None or empty and use final queue
                with torch.no_grad():
                    for input_val, target_val in valid_queue_final:  # Use final queue
                        input_val = input_val.to(device)
                        target_val = target_val.to(device, non_blocking=True)
                        logits_val = arch_to_train(input_val)
                        prec1, _ = utils.accuracy(logits_val, target_val, topk=(1, 5))
                        val_acc_meter.update(prec1.item(), input_val.size(0))

            scheduler.step()
            epoch_end_time = time.time()
            current_epoch_time = epoch_end_time - epoch_start_time
            local_train_time_total += current_epoch_time

            logger.info(
                f"  Arch {i + 1} Epoch {epoch}: TrainAcc={train_acc_meter.avg:.4f}, ValAcc={val_acc_meter.avg:.4f}, Time={current_epoch_time:.2f}s, LR={scheduler.get_last_lr()[0]:.6f}"
            )

            if val_acc_meter.avg > best_val_acc_local:
                best_val_acc_local = val_acc_meter.avg

        arch_to_train.eval()
        test_acc_meter = utils.AverageMeter()
        if (
            test_queue_final
        ):  # Ensure test_queue_final is not None or empty and use final queue
            with torch.no_grad():
                for input_test, target_test in test_queue_final:  # Use final queue
                    input_test = input_test.to(device)
                    target_test = target_test.to(device, non_blocking=True)
                    logits_test = arch_to_train(input_test)
                    prec1, _ = utils.accuracy(logits_test, target_test, topk=(1, 5))
                    test_acc_meter.update(prec1.item(), input_test.size(0))
        final_test_acc_local = test_acc_meter.avg

        logger.info(
            f"  Local Train Finished Arch {i + 1}: TotalTime={local_train_time_total:.2f}s, FinalTrainAcc={final_train_acc_local:.4f}, BestValAcc={best_val_acc_local:.4f}, FinalTestAcc={final_test_acc_local:.4f}"
        )

        comp_factor = -1.0
        if (
            queried_train_time is not None and queried_train_time > 1e-6
        ):  # Avoid division by zero or near-zero
            comp_factor = local_train_time_total / queried_train_time

        current_arch_result = {
            "op_indices": op_indices,
            "queried_train_time": queried_train_time,
            "queried_train_acc": queried_train_acc,
            "queried_val_acc": queried_val_acc,
            "queried_test_acc": queried_test_acc,
            "local_train_time": local_train_time_total,
            "local_train_acc": final_train_acc_local,
            "local_val_acc": best_val_acc_local,
            "local_test_acc": final_test_acc_local,
            "computational_factor": comp_factor,
            "seed_of_run": args.seed,  # Add seed to identify the run
        }
        results.append(current_arch_result)

        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(
            f"Saved/Appended results for architecture {i + 1} to {results_json_path}"
        )

    logger.info(f"Experiment finished. All results saved to {results_json_path}")
    all_factors = [
        r["computational_factor"] for r in results if r["computational_factor"] > 0
    ]
    if all_factors:
        logger.info(
            f"Summary of Computational Factors ({len(all_factors)} architectures):"
        )
        logger.info(f"  Average: {np.mean(all_factors):.4f}")
        logger.info(f"  Std Dev: {np.std(all_factors):.4f}")
        logger.info(f"  Min: {np.min(all_factors):.4f}")
        logger.info(f"  Max: {np.max(all_factors):.4f}")
        logger.info(f"  All factors: {all_factors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Computational Factor for NAS-Bench-201"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--num_archs",
        type=int,
        default=3,
        help="Number of random architectures to test if --arch_indices is not provided.",
    )
    parser.add_argument(
        "--arch_indices",
        type=str,
        default=None,
        help="JSON string of a list of op_indices lists, e.g., '[[0,1,2,3,4,0],[1,2,3,4,0,1]]'. Overrides --num_archs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs for local training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for local training."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="experiments/comp_factor_naslib",
        help="Base output directory.",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=50,
        help="Logging frequency for training steps per epoch.",
    )

    script_args = parser.parse_args()
    main(script_args)
