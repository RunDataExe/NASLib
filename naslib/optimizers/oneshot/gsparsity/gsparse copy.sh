#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=group_sparsity_nas
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/run/group_sparsity_nas.out


python naslib/optimizers/oneshot/gsparsity/runner.py --config-file naslib/optimizers/oneshot/gsparsity/config.yaml


# TODO find out how to time runs in naslib