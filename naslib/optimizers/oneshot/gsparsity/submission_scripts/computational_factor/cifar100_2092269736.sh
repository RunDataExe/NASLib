#!/bin/bash
#SBATCH --time=01:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=comp_factor_cifar100
#SBATCH --output=experiments/comp_factor_naslib/cifar100/slurm_comp_factor_cifar100_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

# Ensure the output directory for SLURM logs exists
mkdir -p naslib/optimizers/oneshot/gsparsity/submission_scripts/computational_factor/cifar100

python naslib/utils/computational_factor.py \
    --dataset cifar100 \
    --num_archs 1 \
    --epochs 200 \
    --batch_size 256 \
    --seed 2092269736 \
    --out_dir naslib/optimizers/oneshot/gsparsity/submission_scripts/computational_factor \
    --log_freq 50