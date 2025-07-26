#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=comp_factor_imagenet16_120
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_training_verification_and_querrybased_computational_factor/slurm_logs/ImageNet16-120/slurm_comp_factor_imagenet16_120_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

# Ensure the output directory for SLURM logs exists
mkdir -p naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_training_verification_and_querrybased_computational_factor/ImageNet16-120

python naslib/utils/computational_factor_nasbench201_querry_archs_vs_selftraining.py \
    --dataset ImageNet16-120 \
    --num_archs 1 \
    --epochs 200 \
    --batch_size 256 \
    --seed 1544457859 \
    --out_dir naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_training_verification_and_querrybased_computational_factor \
    --log_freq 50