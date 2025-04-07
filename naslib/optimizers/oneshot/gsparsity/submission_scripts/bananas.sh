#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=bananas_original
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/run/bananas_original.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/configurator.py \
    --optimizer bananas \
    --search_space nasbench301 \
    --dataset cifar10 \
    --seed 42 \
    --out_dir naslib/optimizers/oneshot/gsparsity/test