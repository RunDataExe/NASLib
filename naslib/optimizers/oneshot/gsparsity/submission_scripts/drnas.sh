#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=drnas_original
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/run/drnas_nas_original.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/configurator.py \
    --optimizer drnas \
    --search_space nasbench301 \
    --dataset cifar10 \
    --seed 42 \
    --out_dir naslib/optimizers/oneshot/gsparsity/test