#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=zc_scoring_timefactor_cifar10
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor/slurm_logs/slurm_comp_factor_cifar10_1544457859_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de



python naslib/optimizers/oneshot/gsparsity/zc_score_archs_time_saver.py \
    --dataset cifar10 \