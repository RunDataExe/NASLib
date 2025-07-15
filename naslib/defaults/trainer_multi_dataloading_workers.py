import codecs
from naslib.search_spaces.core.graph import Graph
from naslib.utils.vis import plot_architectural_weights
import time
import json
import logging
import os
import copy
import torch
import numpy as np
import optuna

from fvcore.common.checkpoint import PeriodicCheckpointer

from naslib.search_spaces.core.query_metrics import Metric

from naslib import utils
from naslib.utils.log import log_every_n_seconds, log_first_n

from typing import Callable
from .additional_primitives import DropPathWrapper

logger = logging.getLogger(__name__)


class Trainer(object):
    """
    Default implementation that handles dataloading and preparing batches, the
    train loop, gathering statistics, checkpointing and doing the final
    final evaluation.

    If this does not fulfil your needs free do subclass it and implement your
    required logic.
    """

    def __init__(self, optimizer, config, lightweight_output=False):
        """
        Initializes the trainer.

        Args:
            optimizer: A NASLib optimizer
            config (AttrDict): The configuration loaded from a yaml file, e.g
                via  `utils.get_config_from_args()`
        """
        self.optimizer = optimizer
        self.config = config
        self.epochs = self.config.search.epochs
        self.lightweight_output = lightweight_output

        # Early stopping
        self.early_stopping_enabled = False
        self.stop_training = False  # Flag to signal immediate stop
        if "early_stopping" in self.config.search:
            self.early_stopping_enabled = True
            es_config = self.config.search.early_stopping
            self.early_stopping_criterion = es_config.criterion
            self.early_stopping_patience = es_config.patience
            self.early_stopping_threshold = float(es_config.threshold)

            # Determine mode automatically based on criterion
            if "loss" in self.early_stopping_criterion:
                self.early_stopping_mode = "min"
            else:  # for acc, runtime
                self.early_stopping_mode = "max"

            self.early_stopping_counter = 0
            self.early_stopping_best_value = (
                float("inf") if self.early_stopping_mode == "min" else float("-inf")
            )
            logger.info(
                f"Early stopping enabled: criterion={self.early_stopping_criterion}, "
                f"patience={self.early_stopping_patience}, threshold={self.early_stopping_threshold}, "
                f"mode={self.early_stopping_mode}"
            )

        # preparations
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  #! originally torch.device("cuda" if torch.cuda.is_available() else "cpu") #? alternative torch.device("cpu")

        # measuring stuff
        self.train_top1 = utils.AverageMeter()
        self.train_top5 = utils.AverageMeter()
        self.train_loss = utils.AverageMeter()
        self.val_top1 = utils.AverageMeter()
        self.val_top5 = utils.AverageMeter()
        self.val_loss = utils.AverageMeter()

        n_parameters = optimizer.get_model_size()
        # logger.info("param size = %fMB", n_parameters)
        self.search_trajectory = utils.AttrDict(
            {
                "train_acc": [],
                "train_loss": [],
                "valid_acc": [],
                "valid_loss": [],
                "test_acc": [],
                "test_loss": [],
                "runtime": [],
                "train_time": [],
                "arch_eval": [],
                "params": n_parameters,
            }
        )

    def search(
        self,
        resume_from="",
        summary_writer=None,
        after_epoch: Callable[[int], None] = None,
        report_incumbent=True,
        trial: optuna.trial.Trial = None,
    ):
        """
        Start the architecture search.

        Generates a json file with training statistics.

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                train from scratch.
        """
        logger.info("Beginning search")

        np.random.seed(self.config.search.seed)
        torch.manual_seed(self.config.search.seed)

        self.optimizer.before_training()
        checkpoint_freq = self.config.search.checkpoint_freq
        if self.optimizer.using_step_function:
            self.scheduler = self.build_search_scheduler(
                self.optimizer.op_optimizer, self.config
            )

            start_epoch = self._setup_checkpointers(
                resume_from, period=checkpoint_freq, scheduler=self.scheduler
            )
        else:
            start_epoch = self._setup_checkpointers(resume_from, period=checkpoint_freq)

        if start_epoch > 0:
            errors_json_path = os.path.join(self.config.save, "errors.json")
            if os.path.exists(errors_json_path):
                logger.info(
                    f"Resuming: Found existing errors.json at {errors_json_path}. Attempting to load previous trajectory."
                )
                try:
                    with codecs.open(errors_json_path, "r", encoding="utf-8") as f:
                        loaded_data = json.load(f)

                    previous_trajectory_data = None
                    if not self.lightweight_output:
                        if isinstance(loaded_data, dict):
                            previous_trajectory_data = loaded_data
                        else:
                            logger.warning(
                                "Resuming: errors.json is not in the expected dict format for non-lightweight output. Skipping trajectory load."
                            )
                    else:  # lightweight_output is True
                        if (
                            isinstance(loaded_data, list)
                            and len(loaded_data) == 2
                            and isinstance(loaded_data[1], dict)
                        ):
                            previous_trajectory_data = loaded_data[
                                1
                            ]  # The trajectory dict
                        else:
                            logger.warning(
                                "Resuming: errors.json is not in the expected list format for lightweight output. Skipping trajectory load."
                            )

                    if previous_trajectory_data:
                        for (
                            key,
                            current_val_in_trajectory,
                        ) in self.search_trajectory.items():
                            if (
                                isinstance(current_val_in_trajectory, list)
                                and key in previous_trajectory_data
                                and isinstance(previous_trajectory_data[key], list)
                            ):
                                # Load data for epochs 0 to start_epoch - 1
                                self.search_trajectory[key] = previous_trajectory_data[
                                    key
                                ][:start_epoch]
                                # self.search_trajectory[key] = previous_trajectory_data[
                                #     key
                                # ]
                                logger.info(
                                    f"Resuming: Loaded {len(self.search_trajectory[key])} entries for trajectory key '{key}' from errors.json (expected up to {start_epoch})."
                                )

                        logger.info(
                            f"Successfully processed previous search_trajectory from errors.json for resuming at epoch {start_epoch}."
                        )

                        # Re-initialize early stopping state from loaded trajectory
                        if self.early_stopping_enabled:
                            self._reinitialize_early_stopping_from_history()

                except Exception as e:
                    logger.error(
                        f"Resuming: Failed to load or parse errors.json: {e}. Starting with a fresh trajectory for metrics."
                    )
            else:
                logger.info(
                    "Resuming: No existing errors.json found. Starting with a fresh trajectory for metrics."
                )

        if self.optimizer.using_step_function:
            self.train_queue, self.valid_queue, _ = self.build_search_dataloaders(
                self.config
            )
            # Preload validation data to avoid multi-worker iterator deadlocks
            # logger.info("Preloading validation data into a list.")
            # valid_data_list = [
            #     (d[0].to(self.device), d[1].to(self.device, non_blocking=True))
            #     for d in self.valid_queue
            # ]
            # logger.info(f"Preloaded {len(valid_data_list)} validation batches.")

        arch_weights = []
        for e in range(start_epoch, self.epochs):
            # Check if early stopping was triggered during re-initialization
            if self.stop_training:
                logger.info(
                    f"Stopping search at epoch {e} because historical data already met the early stopping criteria."
                )
                break

            start_time = time.time()
            self.optimizer.new_epoch(e)

            if self.optimizer.using_step_function:
                valid_iterator = iter(self.valid_queue)
                for step, data_train in enumerate(self.train_queue):
                    if self.config.save_arch_weights is True:
                        if len(arch_weights) == 0:
                            for edge_weights in self.optimizer.architectural_weights:
                                arch_weights.append(
                                    torch.unsqueeze(edge_weights.detach(), dim=0)
                                )
                        else:
                            for i, edge_weights in enumerate(
                                self.optimizer.architectural_weights
                            ):
                                arch_weights[i] = torch.cat(
                                    (
                                        arch_weights[i],
                                        torch.unsqueeze(edge_weights.detach(), dim=0),
                                    ),
                                    dim=0,
                                )

                    data_train = (
                        data_train[0].to(self.device),
                        data_train[1].to(self.device, non_blocking=True),
                    )
                    try:
                        data_val = next(valid_iterator)
                    except StopIteration:
                        valid_iterator = iter(self.valid_queue)
                        data_val = next(valid_iterator)

                    data_val = (
                        data_val[0].to(self.device),
                        data_val[1].to(self.device, non_blocking=True),
                    )
                    stats = self.optimizer.step(data_train, data_val)
                    logits_train, logits_val, train_loss, val_loss = stats

                    self._store_accuracies(logits_train, data_train[1], "train")
                    self._store_accuracies(logits_val, data_val[1], "val")

                    log_every_n_seconds(
                        logging.INFO,
                        "Epoch {}-{}, Train loss: {:.5f}, validation loss: {:.5f}, learning rate: {}".format(
                            e, step, train_loss, val_loss, self.scheduler.get_last_lr()
                        ),
                        n=5,
                    )

                    if torch.cuda.is_available():
                        log_first_n(
                            logging.INFO,
                            "cuda consumption\n {}".format(torch.cuda.memory_summary()),
                            n=3,
                        )

                    self.train_loss.update(float(train_loss.detach().cpu()))
                    self.val_loss.update(float(val_loss.detach().cpu()))

                self.scheduler.step()

                end_time = time.time()

                self.search_trajectory.train_acc.append(self.train_top1.avg)
                self.search_trajectory.train_loss.append(self.train_loss.avg)
                self.search_trajectory.valid_acc.append(self.val_top1.avg)
                self.search_trajectory.valid_loss.append(self.val_loss.avg)
                self.search_trajectory.runtime.append(end_time - start_time)
            else:
                end_time = time.time()
                # TODO: nasbench101 does not have train_loss, valid_loss, test_loss implemented, so this is a quick fix for now
                # train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss = self.optimizer.train_statistics()
                (
                    train_acc,
                    valid_acc,
                    test_acc,
                    train_time,
                ) = self.optimizer.train_statistics(report_incumbent)
                train_loss, valid_loss, test_loss = -1, -1, -1

                self.search_trajectory.train_acc.append(train_acc)
                self.search_trajectory.train_loss.append(train_loss)
                self.search_trajectory.valid_acc.append(valid_acc)
                self.search_trajectory.valid_loss.append(valid_loss)
                self.search_trajectory.test_acc.append(test_acc)
                self.search_trajectory.test_loss.append(test_loss)
                # For query-based methods, calculate runtime based on scaled benchmark time
                comp_factor = getattr(self.config.search, "comp_factor", 1.0)
                logging.info(
                    f"Using computation factor {comp_factor} for scaling runtime."
                )
                scaling_epochs = getattr(
                    self.config.search, "scaling_factor_epochs", 1.0
                )
                logging.info(
                    f"Using scaling factor {scaling_epochs} for scaling runtime."
                )

                # Determine the type of optimizer to apply the correct runtime scaling.
                # Self-training optimizers have a `_train_and_evaluate_arch` method.
                # For meta-optimizers, we check their sub-optimizers.
                is_self_training = hasattr(self.optimizer, "_train_and_evaluate_arch")
                if not is_self_training and hasattr(self.optimizer, "stage1_optimizer"):
                    is_self_training = hasattr(
                        self.optimizer.stage1_optimizer, "_train_and_evaluate_arch"
                    )

                is_real_time = getattr(self.config.search, "use_real_time", False)

                if is_self_training:
                    # For self-training, `train_time` is always the measured wall-clock time.
                    # The `comp_factor` determines if this time is scaled.
                    # - If use_real_time=True, configurator sets comp_factor=1.0 -> unscaled real time.
                    # - If use_real_time=False, configurator loads the actual comp_factor -> scaled real time.
                    scaled_runtime = train_time * comp_factor
                    is_real_time_str = (
                        "unscaled real-time"
                        if comp_factor == 1.0
                        else "scaled real-time"
                    )
                    logging.info(
                        f"Self-training runtime ({is_real_time_str}): "
                        f"measured_time:{train_time:.4f} * comp_factor:{comp_factor:.4f} = scaled_runtime:{scaled_runtime:.4f}"
                    )
                else:
                    # Standard Query-based method. `train_time` is the queried time from the benchmark.
                    scaled_runtime = train_time * scaling_epochs * comp_factor
                    logging.info(
                        f"Standard query-based runtime: train_time:{train_time} * scaling_epochs:{scaling_epochs} * comp_factor:{comp_factor} = scaled_runtime:{scaled_runtime}"
                    )

                self.search_trajectory.runtime.append(scaled_runtime)
                self.search_trajectory.train_time.append(train_time)
                self.train_top1.avg = train_acc
                self.val_top1.avg = valid_acc

            self.periodic_checkpointer.step(e)

            anytime_results = self.optimizer.test_statistics()
            # if anytime_results:
            # record anytime performance
            # self.search_trajectory.arch_eval.append(anytime_results)
            # log_every_n_seconds(
            #     logging.INFO,
            #     "Epoch {}, Anytime results: {}".format(e, anytime_results),
            #     n=5,
            # )

            self._log_to_json()

            # Early stopping check
            if self.early_stopping_enabled:
                if self._check_early_stopping():
                    logger.info(
                        f"Stopping early at epoch {e} due to no improvement for {self.early_stopping_patience} epochs."
                    )
                    break

            # Report to Optuna and check for pruning
            if trial:
                trial.report(
                    self.val_top1.avg, e + 1
                )  # e + 1 because epochs are 0-indexed
                if trial.should_prune():
                    self.optimizer.after_training()
                    raise optuna.TrialPruned()

            self._log_and_reset_accuracies(e, summary_writer)

            if after_epoch is not None:
                after_epoch(e)

        logger.info(
            f"Saving architectural weight tensors: {self.config.save}/arch_weights.pt"
        )
        if hasattr(self.config, "save_arch_weights") and self.config.save_arch_weights:
            torch.save(arch_weights, f"{self.config.save}/arch_weights.pt")
            if (
                hasattr(self.config, "plot_arch_weights")
                and self.config.plot_arch_weights
            ):
                plot_architectural_weights(self.config, self.optimizer)

        self.optimizer.after_training()

        if summary_writer is not None:
            summary_writer.close()

        logger.info("Training finished")

    def evaluate_oneshot(self, resume_from="", dataloader=None):
        """
        Evaluate the one-shot model on the specified dataset.

        Generates a json file with training statistics.

        Args:
            resume_from (str): Checkpoint file to resume from. If not given then
                evaluate with the current one-shot weights.
        """
        logger.info("Start one-shot evaluation")
        self.optimizer.before_training()
        self._setup_checkpointers(resume_from)

        loss = torch.nn.CrossEntropyLoss()

        if dataloader is None:
            # load only the validation data
            _, dataloader, _ = self.build_search_dataloaders(self.config)

        self.optimizer.graph.eval()
        with torch.no_grad():
            start_time = time.time()
            for step, data_val in enumerate(dataloader):
                input_val = data_val[0].to(self.device)
                target_val = data_val[1].to(self.device, non_blocking=True)

                logits_val = self.optimizer.graph(input_val)
                val_loss = loss(logits_val, target_val)

                self._store_accuracies(logits_val, data_val[1], "val")
                self.val_loss.update(float(val_loss.detach().cpu()))

            end_time = time.time()

            self.search_trajectory.valid_acc.append(self.val_top1.avg)
            self.search_trajectory.valid_loss.append(self.val_loss.avg)
            self.search_trajectory.runtime.append(end_time - start_time)

            self._log_to_json()

        logger.info("Evaluation finished")
        return self.val_top1.avg

    def evaluate(
        self,
        retrain: bool = True,
        search_model: str = "",
        resume_from: str = "",
        best_arch: Graph = None,
        dataset_api: object = None,
        metric: Metric = None,
    ):
        """
        Evaluate the final architecture as given from the optimizer.

        If the search space has an interface to a benchmark then query that.
        Otherwise train as defined in the config.

        Args:
            retrain (bool)      : Reset the weights from the architecure search
            search_model (str)  : Path to checkpoint file that was created during search. If not provided,
                                  then try to load 'model_final.pth' from search
            resume_from (str)   : Resume retraining from the given checkpoint file.
            best_arch           : Parsed model you want to directly evaluate and ignore the final model
                                  from the optimizer.
            dataset_api         : Dataset API to use for querying model performance.
            metric              : Metric to query the benchmark for.
        """
        logger.info("Start evaluation")
        if not best_arch:
            if not search_model:
                search_model = os.path.join(
                    self.config.save, "search", "model_final.pth"
                )
            self._setup_checkpointers(search_model)  # required to load the architecture

            best_arch = self.optimizer.get_final_architecture()
        logger.info(f"Final architecture hash: {best_arch.get_hash()}")

        if best_arch.QUERYABLE:
            if metric is None:
                metric = Metric.TEST_ACCURACY
            result = best_arch.query(
                metric=metric, dataset=self.config.dataset, dataset_api=dataset_api
            )
            logger.info("Queried results ({}): {}".format(metric, result))
            return result
        else:
            best_arch.to(self.device)
            if retrain:
                logger.info("Starting retraining from scratch")
                best_arch.reset_weights(inplace=True)

                (
                    self.train_queue,
                    self.valid_queue,
                    self.test_queue,
                ) = self.build_eval_dataloaders(self.config)

                optim = self.build_eval_optimizer(best_arch.parameters(), self.config)
                scheduler = self.build_eval_scheduler(optim, self.config)

                start_epoch = self._setup_checkpointers(
                    resume_from,
                    search=False,
                    period=self.config.evaluation.checkpoint_freq,
                    model=best_arch,  # checkpointables start here
                    optim=optim,
                    scheduler=scheduler,
                )

                grad_clip = self.config.evaluation.grad_clip
                loss = torch.nn.CrossEntropyLoss()

                self.train_top1.reset()
                self.train_top5.reset()
                self.val_top1.reset()
                self.val_top5.reset()

                # Enable drop path
                best_arch.update_edges(
                    update_func=lambda edge: edge.data.set(
                        "op", DropPathWrapper(edge.data.op)
                    ),
                    scope=best_arch.OPTIMIZER_SCOPE,
                    private_edge_data=True,
                )

                # train from scratch
                epochs = self.config.evaluation.epochs
                for e in range(start_epoch, epochs):
                    best_arch.train()

                    if torch.cuda.is_available():
                        log_first_n(
                            logging.INFO,
                            "cuda consumption\n {}".format(torch.cuda.memory_summary()),
                            n=20,
                        )

                    # update drop path probability
                    drop_path_prob = self.config.evaluation.drop_path_prob * e / epochs
                    best_arch.update_edges(
                        update_func=lambda edge: edge.data.set(
                            "drop_path_prob", drop_path_prob
                        ),
                        scope=best_arch.OPTIMIZER_SCOPE,
                        private_edge_data=True,
                    )

                    # Train queue
                    for i, (input_train, target_train) in enumerate(self.train_queue):
                        input_train = input_train.to(self.device)
                        target_train = target_train.to(self.device, non_blocking=True)

                        optim.zero_grad()
                        logits_train = best_arch(input_train)
                        train_loss = loss(logits_train, target_train)
                        if hasattr(
                            best_arch, "auxilary_logits"
                        ):  # darts specific stuff
                            log_first_n(logging.INFO, "Auxiliary is used", n=10)
                            auxiliary_loss = loss(
                                best_arch.auxilary_logits(), target_train
                            )
                            train_loss += (
                                self.config.evaluation.auxiliary_weight * auxiliary_loss
                            )
                        train_loss.backward()
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(
                                best_arch.parameters(), grad_clip
                            )
                        optim.step()

                        self._store_accuracies(logits_train, target_train, "train")
                        log_every_n_seconds(
                            logging.INFO,
                            "Epoch {}-{}, Train loss: {:.5}, learning rate: {}".format(
                                e, i, train_loss, scheduler.get_last_lr()
                            ),
                            n=5,
                        )

                    # Validation queue
                    if self.valid_queue:
                        best_arch.eval()
                        for i, (input_valid, target_valid) in enumerate(
                            self.valid_queue
                        ):
                            input_valid = input_valid.to(self.device).float()
                            target_valid = target_valid.to(self.device).float()

                            # just log the validation accuracy
                            with torch.no_grad():
                                logits_valid = best_arch(input_valid)
                                self._store_accuracies(
                                    logits_valid, target_valid, "val"
                                )

                    scheduler.step()
                    self.periodic_checkpointer.step(e)
                    self._log_and_reset_accuracies(e)

            # Disable drop path
            best_arch.update_edges(
                update_func=lambda edge: edge.data.set(
                    "op", edge.data.op.get_embedded_ops()
                ),
                scope=best_arch.OPTIMIZER_SCOPE,
                private_edge_data=True,
            )

            # measure final test accuracy
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()

            best_arch.eval()

            for i, data_test in enumerate(self.test_queue):
                input_test, target_test = data_test
                input_test = input_test.to(self.device)
                target_test = target_test.to(self.device, non_blocking=True)

                n = input_test.size(0)

                with torch.no_grad():
                    logits = best_arch(input_test)

                    prec1, prec5 = utils.accuracy(logits, target_test, topk=(1, 5))
                    top1.update(prec1.data.item(), n)
                    top5.update(prec5.data.item(), n)

                log_every_n_seconds(
                    logging.INFO,
                    "Inference batch {} of {}.".format(i, len(self.test_queue)),
                    n=5,
                )

            logger.info(
                "Evaluation finished. Test accuracies: top-1 = {:.5}, top-5 = {:.5}".format(
                    top1.avg, top5.avg
                )
            )

            return top1.avg

    @staticmethod
    def build_search_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode="train"
        )
        return train_queue, valid_queue, _  # test_queue is not used in search currently

    @staticmethod
    def build_eval_dataloaders(config):
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode="val"
        )
        return train_queue, valid_queue, test_queue

    @staticmethod
    def build_eval_optimizer(parameters, config):
        return torch.optim.SGD(
            parameters,
            lr=config.evaluation.learning_rate,
            momentum=config.evaluation.momentum,
            weight_decay=config.evaluation.weight_decay,
        )

    @staticmethod
    def build_search_scheduler(optimizer, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.search.epochs,
            eta_min=config.search.learning_rate_min,
        )

    @staticmethod
    def build_eval_scheduler(optimizer, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.evaluation.epochs,
            eta_min=config.evaluation.learning_rate_min,
        )

    def _log_and_reset_accuracies(self, epoch, writer=None):
        logger.info(
            "Epoch {} done. Train accuracy: {:.5f}, Validation accuracy: {:.5f}".format(
                epoch,
                self.train_top1.avg,
                self.val_top1.avg,
            )
        )

        if writer is not None:
            writer.add_scalar("Train accuracy (top 1)", self.train_top1.avg, epoch)
            writer.add_scalar("Train accuracy (top 5)", self.train_top5.avg, epoch)
            writer.add_scalar("Train loss", self.train_loss.avg, epoch)
            writer.add_scalar("Validation accuracy (top 1)", self.val_top1.avg, epoch)
            writer.add_scalar("Validation accuracy (top 5)", self.val_top5.avg, epoch)
            writer.add_scalar("Validation loss", self.val_loss.avg, epoch)

        self.train_top1.reset()
        self.train_top5.reset()
        self.train_loss.reset()
        self.val_top1.reset()
        self.val_top5.reset()
        self.val_loss.reset()

    def _store_accuracies(self, logits, target, split):
        """Update the accuracy counters"""
        logits = logits.clone().detach().cpu()
        target = target.clone().detach().cpu()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = logits.size(0)

        if split == "train":
            self.train_top1.update(prec1.data.item(), n)
            self.train_top5.update(prec5.data.item(), n)
        elif split == "val":
            self.val_top1.update(prec1.data.item(), n)
            self.val_top5.update(prec5.data.item(), n)
        else:
            raise ValueError("Unknown split: {}. Expected either 'train' or 'val'")

    def _prepare_dataloaders(self, config, mode="train"):
        """
        Prepare train, validation, and test dataloaders with the splits defined
        in the config.

        Args:
            config (AttrDict): config from config file.
        """
        train_queue, valid_queue, test_queue, _, _ = utils.get_train_val_loaders(
            config, mode
        )
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.test_queue = test_queue

    def _setup_checkpointers(
        self, resume_from="", search=True, period=1, **add_checkpointables
    ):
        """
        Sets up a periodic chechkpointer which can be used to save checkpoints
        at every epoch. It will call optimizer's `get_checkpointables()` as objects
        to store.

        Args:
            resume_from (str): A checkpoint file to resume the search or evaluation from.
            search (bool): Whether search or evaluation phase is checkpointed. This is required
                because the files are in different folders to not be overridden
            add_checkpointables (object): Additional things to checkpoint together with the
                optimizer's checkpointables.
        """
        checkpointables = self.optimizer.get_checkpointables()
        checkpointables.update(add_checkpointables)

        checkpointer = utils.Checkpointer(
            model=checkpointables.pop("model"),
            save_dir=self.config.save + "/search"
            if search
            else self.config.save + "/eval",
            # **checkpointables #NOTE: this is throwing an Error
        )

        self.periodic_checkpointer = PeriodicCheckpointer(
            checkpointer,
            period=period,
            max_iter=self.config.search.epochs
            if search
            else self.config.evaluation.epochs,
        )

        if resume_from:
            logger.info("loading model from file {}".format(resume_from))
            checkpoint = checkpointer.resume_or_load(resume_from, resume=True)
            if checkpointer.has_checkpoint():
                return checkpoint.get("iteration", -1) + 1
        return 0

    def _reinitialize_early_stopping_from_history(self):
        """
        Recalculates early stopping state (best_value, counter) from the loaded
        search trajectory. This is crucial for correct behavior when resuming.
        """
        logger.info("Re-initializing early stopping state from historical data.")
        trajectory = self.search_trajectory.get(self.early_stopping_criterion, [])
        if not trajectory:
            logger.warning(
                "Early stopping history re-initialization: Trajectory is empty. Nothing to do."
            )
            return

        # Reset to initial state before recalculating
        self.early_stopping_counter = 0
        self.early_stopping_best_value = (
            float("inf") if self.early_stopping_mode == "min" else float("-inf")
        )

        for value in trajectory:
            # Ignore placeholder values that might come from some query-based methods
            if value in [-1, None]:
                continue

            if self.early_stopping_mode == "min":
                improvement = self.early_stopping_best_value - value
            else:  # max
                improvement = value - self.early_stopping_best_value

            if improvement > self.early_stopping_threshold:
                self.early_stopping_best_value = value
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1

        logger.info(
            f"Early stopping state re-initialized: Best value = {self.early_stopping_best_value}, "
            f"Patience counter = {self.early_stopping_counter}"
        )

        # Check if we should stop immediately based on the loaded history
        if self.early_stopping_counter >= self.early_stopping_patience:
            logger.warning(
                f"Historical data meets early stopping criteria (Patience: {self.early_stopping_patience}). "
                "Training will be stopped before the next epoch."
            )
            self.stop_training = True

    def _check_early_stopping(self):
        """Checks if the early stopping condition is met."""
        try:
            current_value = self.search_trajectory[self.early_stopping_criterion][-1]
        except (KeyError, IndexError):
            logger.warning(
                f"Early stopping criterion '{self.early_stopping_criterion}' not found in search_trajectory. Skipping check."
            )
            return

        if self.early_stopping_mode == "min":
            improvement = self.early_stopping_best_value - current_value
        else:  # max
            improvement = current_value - self.early_stopping_best_value

        if improvement > self.early_stopping_threshold:
            self.early_stopping_best_value = current_value
            self.early_stopping_counter = 0
            logger.info(
                f"Early stopping: New best value for {self.early_stopping_criterion}: {current_value:.6f}"
            )
        else:
            self.early_stopping_counter += 1
            logger.info(
                f"Early stopping: No improvement for {self.early_stopping_counter} epochs. "
                f"Best {self.early_stopping_criterion}: {self.early_stopping_best_value:.6f}, "
                f"Current: {current_value:.6f}"
            )

        return self.early_stopping_counter >= self.early_stopping_patience

    def _log_to_json(self):
        """log training statistics to json file"""
        if not os.path.exists(self.config.save):
            os.makedirs(self.config.save)
        if not self.lightweight_output:
            with codecs.open(
                os.path.join(self.config.save, "errors.json"), "w", encoding="utf-8"
            ) as file:
                json.dump(self.search_trajectory, file, separators=(",", ":"))
        else:
            with codecs.open(
                os.path.join(self.config.save, "errors.json"), "w", encoding="utf-8"
            ) as file:
                lightweight_dict = copy.deepcopy(self.search_trajectory)
                for key in ["arch_eval", "train_loss", "valid_loss", "test_loss"]:
                    lightweight_dict.pop(key)
                json.dump([self.config, lightweight_dict], file, separators=(",", ":"))
