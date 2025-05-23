from naslib.search_spaces import NasBench201SearchSpace
import torch
from naslib.utils import get_train_val_loaders
from fvcore.common.config import CfgNode
import json
from naslib.optimizers.oneshot.gsparsity.shape_optimizer import GSparseOptimizer
from naslib.optimizers.oneshot.gsparsity.ProxSGD_for_groups import ProxSGD
from naslib.utils.shape_tracker import ShapeTracker

# Configuration
config_dict = {
    'data': "naslib/data",
    'dataset': 'cifar100', # cifar10, cifar100, ImageNet16-120
    'batch_size': 64,
    'num_workers': 4,
    'train_portion': 0.8,
    'search': {
        'seed': 42,
        'batch_size': 64,
        'learning_rate': 0.025,
        'momentum': 0.9,
        'weight_decay': 0.0003,
        'grad_clip': 5.0,
        'threshold': 0.001,
        'normalization': True,
        'normalization_exponent': 0.5,
        'cutout': False
    }
}

config = CfgNode.load_cfg(json.dumps(config_dict))
if config['dataset'] == 'cifar10':
    num_classes = 10
elif config['dataset'] == 'cifar100':
    num_classes = 100
elif config['dataset'] == 'ImageNet16-120':
    num_classes = 120
search_space = NasBench201SearchSpace(n_classes=num_classes)

# Create the GSparseOptimizer and adapt search space
optimizer = GSparseOptimizer(
    config=config,
    op_optimizer=ProxSGD,
    op_optimizer_evaluate=torch.optim.SGD,
    loss_criteria=torch.nn.CrossEntropyLoss()
)
optimizer.adapt_search_space(search_space)
supernet = optimizer.graph

# Create shape tracker and register hooks
tracker = ShapeTracker()
tracker.register_hooks(supernet)

# Load data and run forward pass
train_loader, _, _, _, _ = get_train_val_loaders(config)
inputs, targets = next(iter(train_loader))
print(f"Input tensor shape: {inputs.shape}")

# Forward pass through the supernet
logits = supernet(inputs)
print(f"Output logits shape: {logits.shape}")

# Get shape information
shape_info = tracker.get_shape_info()
# print(shape_info)

# # Save the shape information to a file
# with open(f'Nasbench201_search_space_{config.dataset}_{config.search.batch_size}_shape_info.json', 'w') as f:
#     json.dump(shape_info, f, indent=4)

# Clean up hooks
tracker.clear_hooks()