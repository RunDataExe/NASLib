#!/bin/bash
#SBATCH --time=28:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=group_sparsity_nas/gs_original_non_querryable
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/run/group_sparsity_nas_original_non_querryable.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/configurator_non_querryable.py \
    --optimizer gsparsity \
    --search_space nasbench301 \
    --dataset cifar10 \
    --seed 42 \
    --out_dir naslib/optimizers/oneshot/gsparsity/test_non_querryable