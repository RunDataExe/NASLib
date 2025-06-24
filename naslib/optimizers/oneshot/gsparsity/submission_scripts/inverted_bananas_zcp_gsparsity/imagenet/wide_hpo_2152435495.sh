#!/bin/bash
#SBATCH --time=5:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=wide_hpo_inverted_bananas_zcp_gsparsity_imagenet_2152435495
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/results/inverted_bananas_zcp_gsparsity/imagenet/slurm/wide_hpo_2152435495_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/wide_hpo_search_space_configurator_loss_based.py \
    --optimizer inverted_bananas_zcp_gsparsity \
    --search_space nasbench201 \
    --dataset imagenet \
    --seed 2152435495 \
    --out_dir naslib/optimizers/oneshot/gsparsity/results/inverted_bananas_zcp_gsparsity/imagenet/wide_hpo \
