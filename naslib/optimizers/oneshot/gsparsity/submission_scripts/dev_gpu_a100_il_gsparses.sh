#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dev_gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=h100_group_sparsity_nas/gs_original
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/run/group_sparsity_nas_original_dev_gpu_a100_il.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/configurator.py \
    --optimizer gsparsity \
    --search_space nasbench301 \
    --dataset cifar10 \
    --seed 45 \
    --out_dir naslib/optimizers/oneshot/gsparsity/test