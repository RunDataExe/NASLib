#!/bin/bash
#SBATCH --time=5:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=wide_hpo_gsparsity_cifar10_2152435495
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/results/gsparsity/cifar10/slurm/wide_hpo_2152435495_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/wide_hpo_search_space_configurator_loss_based.py \
    --optimizer gsparsity \
    --search_space nasbench201 \
    --dataset cifar10 \
    --seed 2152435495 \
    --out_dir naslib/optimizers/oneshot/gsparsity/results/gsparsity/cifar10/wide_hpo \
