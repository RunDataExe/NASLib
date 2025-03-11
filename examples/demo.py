import logging
import sys

from naslib.defaults.trainer import Trainer
from naslib.optimizers import (
    DARTSOptimizer,
    GDASOptimizer,
    DrNASOptimizer,
    RandomSearch,
    RegularizedEvolution,
    LocalSearch,
    Bananas,
    BasePredictor,
)

from naslib.search_spaces import (
    NasBench301SearchSpace,
    SimpleCellSearchSpace,
    NasBench201SearchSpace,
    HierarchicalSearchSpace,
)



# from naslib.search_spaces.nasbench101 import graph
from naslib import utils
from naslib.utils import setup_logger

# Read args and config, setup logger
config = utils.get_config_from_args()
utils.set_seed(config.seed)

logger = setup_logger(config.save + "/log.log")
# logger.setLevel(logging.INFO)   # default DEBUG is very verbose

utils.log_args(config)

supported_optimizers = {
    # "darts": DARTSOptimizer(config),
    "darts": DARTSOptimizer(),
    "gdas": GDASOptimizer(config),
    # "drnas": DrNASOptimizer(config),
    "drnas": DrNASOptimizer(),
    "rs": RandomSearch(config),
    "re": RegularizedEvolution(config),
    "ls": LocalSearch(config),
    "bananas": Bananas(config),
    "bp": BasePredictor(config),
}

# Changing the search space is one line of code
# search_space = SimpleCellSearchSpace() #! did not work with this thus error has to be in it or related to it
# search_space = graph.NasBench101SearchSpace()
# search_space = HierarchicalSearchSpace()
# search_space = NasBench301SearchSpace() #! had memmory error
search_space = NasBench201SearchSpace() #! did work

# Changing the optimizer is one line of code
# optimizer = supported_optimizers[config.optimizer]

# import pudb
# pudb.set_trace()

optimizer = supported_optimizers["drnas"]
# optimizer.adapt_search_space(search_space=search_space, dataset=config.dataset) #!


from naslib.utils import get_dataset_api
dataset_api = get_dataset_api(config.search_space, config.dataset)
optimizer.adapt_search_space(search_space=search_space, dataset=config.dataset)
# optimizer.adapt_search_space(search_space, dataset="naslib/data/nb201_cifar10_full_training.pickle")

# optimizer.adapt_search_space(search_space, dataset="cifar10")

# Start the search and evaluation
trainer = Trainer(optimizer=optimizer, config=config)

#! api probably in evaluation called

import pudb

if not config.eval_only:
    checkpoint = utils.get_last_checkpoint(config) if config.resume else ""
    pudb.set_trace()
    trainer.search(resume_from=checkpoint)

#! currently try to verify if checkpoint works -> setted num epochs to 0 such that it instantly evaluates

#! veryfiy if api / evaluation works -> config.eval_only = True

checkpoint = utils.get_last_checkpoint(config, search=False) if config.resume else ""
trainer.evaluate(dataset_api=dataset_api, )
