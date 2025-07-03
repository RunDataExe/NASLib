#!/bin/bash
#SBATCH --time=09:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=stib_imagenet_1544457859
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/submission_scripts/self_training_bananas_verification_and_computational_factor/slurm_logs/ImageNet16-120/slurm_stib_imagenet_1544457859_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/configurator.py \
    --optimizer self_training_inverted_bananas \
    --search_space nasbench201 \
    --dataset ImageNet16-120 \
    --seed 1544457859 \
    --out_dir naslib/optimizers/oneshot/gsparsity/submission_scripts/self_training_bananas_verification_and_computational_factor \
    --search_epochs 3 \
    --eval_epochs 1 \
    --resume True

# For a fair comparison, ensure the following hyperparameters are set in the configurator.py:
# "self_training_inverted_bananas": {
#     "search": {
#         "checkpoint_freq": 1,
#         "epochs": search_epochs,
#         "k": 10,
#         "num_init": 10,
#         "num_ensemble": 5,
#         "predictor_type": "mlp",
#         "acq_fn_type": "its",
#         "acq_fn_optimization": "mutation",
#         "encoding_type": "path",
#         "num_arches_to_mutate": 1,
#         "max_mutations": 1,
#         "num_candidates": 100,
#         "train_epochs": 200,
#         "use_real_time": True,
#     },
# }
# "inverted_bananas": {
#     "search": {
#         "checkpoint_freq": 1,
#         "epochs": search_epochs,
#         "k": 10,
#         "num_init": 10,
#         "num_ensemble": 5,
#         "predictor_type": "mlp",
#         "acq_fn_type": "its",
#         "acq_fn_optimization": "mutation",
#         "encoding_type": "path",
#         "num_arches_to_mutate": 1,
#         "max_mutations": 1,
#         "num_candidates": 100,
#     },
# }