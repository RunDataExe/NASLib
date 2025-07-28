#!/bin/bash
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=comp_factor_cifar10_s2152435495
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_training_verification_and_querrybased_computational_factor/slurm_logs/cifar10/slurm_comp_factor_cifar10_s2152435495_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

# Ensure the output directory for SLURM logs exists
mkdir -p naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_training_verification_and_querrybased_computational_factor/cifar10

python naslib/utils/computational_factor_nasbench201_querry_archs_vs_selftraining.py \
    --dataset cifar10 \
    --num_archs 1 \
    --epochs 200 \
    --batch_size 256 \
    --seed 2152435495 \
    --out_dir naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_training_verification_and_querrybased_computational_factor \
    --log_freq 50