import collections
import logging
import torch
import copy
import numpy as np
import time
from torch.utils.data import Subset, DataLoader
from torch.optim.sgd import SGD
from fvcore.common.config import CfgNode
import json

from naslib.optimizers.core.metaclasses import MetaOptimizer
from naslib.optimizers.discrete.bananas.acquisition_functions import (
    acquisition_function,
)

from naslib.predictors.ensemble import Ensemble
from naslib.predictors.zerocost import ZeroCost
from naslib.predictors.utils.encodings import encode_spec

from naslib.search_spaces.core.query_metrics import Metric

from naslib.utils import (
    AttrDict,
    count_parameters_in_MB,
    get_train_val_loaders,
    get_project_root,
)
from naslib.utils.log import log_every_n_seconds
from naslib.utils.dataset import WorkerInitializer
from naslib import utils

logger = logging.getLogger(__name__)


# This configuration is based on the one in computational_factor.py to ensure
# that the internal training of architectures follows the NAS-Bench-201 paper's setup.
# It is kept internal to this optimizer to avoid conflicts with other configurations,
# especially in two-stage optimizers.
_COMP_FACTOR_CONFIG_TEMPLATE = {
    "dataset": "cifar10",
    "data": str(get_project_root() / "data"),
    "search_space": "nasbench201",
    "optimizer": "eval_only",
    "seed": 42,
    "out_dir": "comp_factor_exp",
    "search": {
        "seed": 42,
        "epochs": 1,
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
        "drop_path_prob": 0.0,
        "cutout": False,
        "cutout_length": 16,
        "cutout_prob": 1.0,
        "train_portion": 0.5,
    },
}


class Bananas(MetaOptimizer):
    # training the models is not implemented
    using_step_function = False

    def __init__(self, config, zc_api=None):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        # New attributes for real training
        self.train_epochs = (
            config.search.train_epochs
            if hasattr(config.search, "train_epochs")
            else 200
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a dedicated config for internal training
        self.train_config = CfgNode.load_cfg(json.dumps(_COMP_FACTOR_CONFIG_TEMPLATE))
        # Inherit dataset, seed, and data path from the main config
        self.train_config.dataset = config.dataset
        self.train_config.seed = config.search.seed
        self.train_config.data = config.data
        # Set epochs for internal training
        self.train_config.evaluation.epochs = self.train_epochs
        # Allow overriding batch size from the main config's search section
        if hasattr(config.search, "batch_size"):
            self.train_config.evaluation.batch_size = config.search.batch_size
        self.train_config.search.batch_size = self.train_config.evaluation.batch_size
        self.train_config.search.seed = self.train_config.seed

        self.train_queue = None
        self.valid_queue = None
        self.test_queue = None
        self.n_classes = None

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.k = config.search.k
        self.num_init = config.search.num_init
        self.num_ensemble = config.search.num_ensemble
        self.predictor_type = config.search.predictor_type
        self.acq_fn_type = config.search.acq_fn_type
        self.acq_fn_optimization = config.search.acq_fn_optimization
        self.encoding_type = config.search.encoding_type  # currently not implemented
        self.num_arches_to_mutate = config.search.num_arches_to_mutate
        self.max_mutations = config.search.max_mutations
        self.num_candidates = config.search.num_candidates
        self.max_zerocost = 1000

        self.train_data = []
        self.next_batch = []
        self.history = torch.nn.ModuleList()

        self.zc = config.search.zc if hasattr(config.search, "zc") else None
        self.semi = "semi" in self.predictor_type
        self.zc_api = zc_api
        self.use_zc_api = (
            config.search.use_zc_api if hasattr(config.search, "use_zc_api") else False
        )
        self.zc_names = (
            config.search.zc_names if hasattr(config.search, "zc_names") else None
        )
        self.zc_only = (
            config.search.zc_only if hasattr(config.search, "zc_only") else False
        )

        self.load_labeled = (
            config.search.load_labeled
            if hasattr(config.search, "load_labeled")
            else False
        )

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        # This optimizer now handles its own training, so the search space does not need to be queryable.
        # assert search_space.QUERYABLE, (
        #     "Bananas is currently only implemented for benchmarks."
        # )

        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api
        self.ss_type = self.search_space.get_type()

        # Setup dataloaders for training, mimicking computational_factor.py
        # This uses the logic from the NAS-Bench-201 paper for data splits.
        if self.train_config.dataset == "cifar10":
            self.n_classes = 10
            self.train_config.evaluation.train_portion = 0.5
            self.train_config.search.train_portion = 0.5
        elif self.train_config.dataset == "cifar100":
            self.n_classes = 100
            self.train_config.evaluation.train_portion = 1.0
            self.train_config.search.train_portion = 1.0
        elif self.train_config.dataset == "ImageNet16-120":
            self.n_classes = 120
            self.train_config.evaluation.train_portion = 1.0
            self.train_config.search.train_portion = 1.0
        else:
            raise ValueError(f"Unsupported dataset: {self.train_config.dataset}")

        train_queue_naslib, valid_queue_naslib, test_queue_naslib, _, _ = (
            get_train_val_loaders(self.train_config, mode="val")
        )

        train_init_fn = WorkerInitializer(self.train_config.seed, is_train=True)
        val_test_init_fn = WorkerInitializer(self.train_config.seed, is_train=False)

        self.train_queue = DataLoader(
            dataset=train_queue_naslib.dataset,
            sampler=train_queue_naslib.sampler,
            batch_size=self.train_config.evaluation.batch_size,
            num_workers=0,  # Use a single thread as requested
            pin_memory=True,
            worker_init_fn=train_init_fn,
        )

        if self.train_config.dataset == "cifar10":
            self.valid_queue = DataLoader(
                dataset=valid_queue_naslib.dataset,
                sampler=valid_queue_naslib.sampler,
                batch_size=self.train_config.evaluation.batch_size,
                num_workers=0,
                pin_memory=True,
                worker_init_fn=val_test_init_fn,
            )
            self.test_queue = DataLoader(
                dataset=test_queue_naslib.dataset,
                sampler=test_queue_naslib.sampler,
                batch_size=self.train_config.evaluation.batch_size,
                num_workers=0,
                pin_memory=True,
                worker_init_fn=val_test_init_fn,
            )
        elif self.train_config.dataset in ["cifar100", "ImageNet16-120"]:
            original_test_set = test_queue_naslib.dataset
            num_original_test = len(original_test_set)
            paper_val_size = 5000 if self.train_config.dataset == "cifar100" else 3000
            np.random.seed(self.train_config.seed)
            indices = np.random.permutation(num_original_test)
            val_indices, test_indices = (
                indices[:paper_val_size],
                indices[paper_val_size:],
            )
            self.valid_queue = DataLoader(
                Subset(original_test_set, val_indices),
                batch_size=self.train_config.evaluation.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                worker_init_fn=val_test_init_fn,
            )
            self.test_queue = DataLoader(
                Subset(original_test_set, test_indices),
                batch_size=self.train_config.evaluation.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                worker_init_fn=val_test_init_fn,
            )
        else:
            raise ValueError(
                f"Dataset {self.train_config.dataset} splitting not implemented."
            )

        logger.info("BANANAS: Dataloaders for real training are set up.")

        if self.zc:
            self.train_loader = self.train_queue
        if self.semi:
            self.unlabeled = []

    def _train_and_evaluate_arch(self, arch):
        """
        Trains a single architecture from scratch using the NAS-Bench-201 training regime.
        """
        if self.search_space.instantiate_model:
            arch.parse()
        arch.to(self.device)
        arch.reset_weights(inplace=True)

        optimizer = SGD(
            arch.parameters(),
            lr=self.train_config.evaluation.learning_rate,
            momentum=self.train_config.evaluation.momentum,
            weight_decay=self.train_config.evaluation.weight_decay,
            nesterov=True,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_epochs,
            eta_min=self.train_config.evaluation.learning_rate_min,
        )
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        total_train_time = 0
        final_train_acc = 0
        best_val_acc = 0

        for epoch in range(self.train_epochs):
            epoch_start_time = time.time()
            arch.train()
            train_acc_meter = utils.AverageMeter()

            for input_train, target_train in self.train_queue:
                input_train, target_train = (
                    input_train.to(self.device),
                    target_train.to(self.device, non_blocking=True),
                )
                optimizer.zero_grad()
                logits_train = arch(input_train)
                loss = criterion(logits_train, target_train)
                loss.backward()
                if self.train_config.evaluation.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        arch.parameters(), self.train_config.evaluation.grad_clip
                    )
                optimizer.step()
                prec1, _ = utils.accuracy(logits_train, target_train, topk=(1, 5))
                train_acc_meter.update(prec1.item(), input_train.size(0))

            final_train_acc = train_acc_meter.avg

            arch.eval()
            val_acc_meter = utils.AverageMeter()
            with torch.no_grad():
                for input_val, target_val in self.valid_queue:
                    input_val, target_val = (
                        input_val.to(self.device),
                        target_val.to(self.device, non_blocking=True),
                    )
                    logits_val = arch(input_val)
                    prec1, _ = utils.accuracy(logits_val, target_val, topk=(1, 5))
                    val_acc_meter.update(prec1.item(), input_val.size(0))

            if val_acc_meter.avg > best_val_acc:
                best_val_acc = val_acc_meter.avg

            scheduler.step()
            total_train_time += time.time() - epoch_start_time
            logger.info(
                f"BANANAS internal training: Arch {arch.get_hash()[:6]} | Epoch {epoch + 1}/{self.train_epochs} | TrainAcc: {train_acc_meter.avg:.4f} | ValAcc: {val_acc_meter.avg:.4f}"
            )

        arch.eval()
        test_acc_meter = utils.AverageMeter()
        with torch.no_grad():
            for input_test, target_test in self.test_queue:
                input_test, target_test = (
                    input_test.to(self.device),
                    target_test.to(self.device, non_blocking=True),
                )
                logits_test = arch(input_test)
                prec1, _ = utils.accuracy(logits_test, target_test, topk=(1, 5))
                test_acc_meter.update(prec1.item(), input_test.size(0))

        return final_train_acc, best_val_acc, test_acc_meter.avg, total_train_time

    def get_zero_cost_predictors(self):
        return {zc_name: ZeroCost(method_type=zc_name) for zc_name in self.zc_names}

    def query_zc_scores(self, arch):
        zc_scores = {}
        zc_methods = self.get_zero_cost_predictors()
        arch_hash = arch.get_hash()
        for zc_name, zc_method in zc_methods.items():
            if self.use_zc_api and str(arch_hash) in self.zc_api:
                score = self.zc_api[str(arch_hash)][zc_name]["score"]
            else:
                zc_method.train_loader = copy.deepcopy(self.train_loader)
                score = zc_method.query(arch, dataloader=zc_method.train_loader)

            if float("-inf") == score:
                score = -1e9
            elif float("inf") == score:
                score = 1e9

            zc_scores[zc_name] = score

        return zc_scores

    def _set_scores(self, model):
        logger.info(f"Starting real training for arch {model.arch_hash[:8]}...")
        train_acc, val_acc, test_acc, train_time = self._train_and_evaluate_arch(
            model.arch
        )
        logger.info(
            f"Finished real training. Val acc: {val_acc:.4f}, Time: {train_time:.2f}s"
        )

        model.accuracy = val_acc  # Used by BANANAS' history
        # Store other metrics for train_statistics()
        model.train_acc = train_acc
        model.test_acc = test_acc
        model.train_time = train_time

        if self.zc and len(self.train_data) <= self.max_zerocost:
            model.zc_scores = self.query_zc_scores(model.arch)

        self.train_data.append(model)
        self._update_history(model)

    def _sample_new_model(self):
        model = torch.nn.Module()
        model.arch = self.search_space.clone()
        model.arch.sample_random_architecture(
            dataset_api=self.dataset_api, load_labeled=self.load_labeled
        )
        model.arch_hash = model.arch.get_hash()

        if self.search_space.instantiate_model == True:
            model.arch.parse()

        return model

    def _get_train(self):
        xtrain = [m.arch for m in self.train_data]
        ytrain = [m.accuracy for m in self.train_data]
        return xtrain, ytrain

    def _get_ensemble(self):
        ensemble = Ensemble(
            num_ensemble=self.num_ensemble,
            ss_type=self.ss_type,
            predictor_type=self.predictor_type,
            zc=self.zc,
            zc_only=self.zc_only,
            config=self.config,
        )

        return ensemble

    def _get_new_candidates(self, ytrain):
        # optimize the acquisition function to output k new architectures
        candidates = []
        if self.acq_fn_optimization == "random_sampling":
            for _ in range(self.num_candidates):
                model = self._sample_new_model()
                # Do not evaluate here. Evaluation happens in the main BO loop.
                candidates.append(model)

        elif self.acq_fn_optimization == "mutation":
            # mutate the k best architectures by x
            best_arch_indices = np.argsort(ytrain)[-self.num_arches_to_mutate :]
            best_archs = [self.train_data[i].arch for i in best_arch_indices]
            candidates = []
            for arch in best_archs:
                for _ in range(
                    int(self.num_candidates / len(best_archs) / self.max_mutations)
                ):
                    candidate = arch.clone()
                    for __ in range(int(self.max_mutations)):
                        arch = self.search_space.clone()
                        arch.mutate(candidate, dataset_api=self.dataset_api)
                        if self.search_space.instantiate_model == True:
                            arch.parse()
                        candidate = arch

                    model = torch.nn.Module()
                    model.arch = candidate
                    model.arch_hash = candidate.get_hash()
                    candidates.append(model)

        else:
            logger.info(
                "{} is not yet supported as a acq fn optimizer".format(
                    self.encoding_type
                )
            )
            raise NotImplementedError()

        return candidates

    def new_epoch(self, epoch):
        if epoch < self.num_init:
            model = self._sample_new_model()
            self._set_scores(model)
        else:
            if len(self.next_batch) == 0:
                # train a neural predictor
                xtrain, ytrain = self._get_train()
                ensemble = self._get_ensemble()

                if self.semi:
                    # create unlabeled data and pass it to the predictor
                    while len(self.unlabeled) < len(xtrain):
                        model = self._sample_new_model()

                        if self.zc and len(self.train_data) <= self.max_zerocost:
                            model.zc_scores = self.query_zc_scores(model.arch)

                        self.unlabeled.append(model)

                    ensemble.set_pre_computations(
                        unlabeled=[m.arch for m in self.unlabeled]
                    )

                if self.zc and len(self.train_data) <= self.max_zerocost:
                    # pass the zero-cost scores to the predictor
                    train_info = {
                        "zero_cost_scores": [m.zc_scores for m in self.train_data]
                    }
                    ensemble.set_pre_computations(xtrain_zc_info=train_info)

                    if self.semi:
                        unlabeled_zc_info = {
                            "zero_cost_scores": [m.zc_scores for m in self.unlabeled]
                        }
                        ensemble.set_pre_computations(
                            unlabeled_zc_info=unlabeled_zc_info
                        )

                ensemble.fit(xtrain, ytrain)

                # define an acquisition function
                acq_fn = acquisition_function(
                    ensemble=ensemble, ytrain=ytrain, acq_fn_type=self.acq_fn_type
                )

                # optimize the acquisition function to output k new architectures
                candidates = self._get_new_candidates(ytrain=ytrain)

                self.next_batch = self._get_best_candidates(candidates, acq_fn)

            # train the next architecture chosen by the neural predictor
            model = self.next_batch.pop()
            self._set_scores(model)

    def _get_best_candidates(self, candidates, acq_fn):
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
        choices = [candidates[i] for i in sorted_indices[-self.k :]]

        return choices

    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self, report_incumbent=True):
        if report_incumbent:
            # self.history contains model objects. get_final_architecture returns the arch.
            # We need the model object.
            if not self.history:
                return -1, -1, -1, -1
            best_model = max(self.history, key=lambda x: x.accuracy)
        else:
            if not self.train_data:
                return -1, -1, -1, -1
            best_model = self.train_data[-1]

        # The model object now has the stats stored from _set_scores
        train_acc = getattr(best_model, "train_acc", -1)
        val_acc = getattr(best_model, "accuracy", -1)  # accuracy is the val_acc
        test_acc = getattr(best_model, "test_acc", -1)
        train_time = getattr(best_model, "train_time", -1)

        if self.search_space.space_name != "nasbench301":
            return (
                train_acc,
                val_acc,
                test_acc,
                train_time,
            )
        else:
            return (
                -1,
                val_acc,
                test_acc,
                train_time,
            )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        if self.search_space.space_name != "nasbench301":
            return best_arch.query(
                Metric.RAW, self.dataset, dataset_api=self.dataset_api
            )
        else:
            return -1

    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {"model": self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)

    def get_arch_as_string(self, arch):
        if self.search_space.get_type() == "nasbench301":
            str_arch = str(list((list(arch[0]), list(arch[1]))))
        else:
            str_arch = str(arch)
        return str_arch
