#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=opzcp_precompute_verify_nb201
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_op_scoring_timefactor/slurm_logs/slurm_precompute_verify_%A_%a.out
#SBATCH --array=0-5
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

set -euo pipefail

mkdir -p naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_op_scoring_timefactor/slurm_logs

DATASETS=("cifar100" "ImageNet16-120")
METHODS=("params" "jacov" "synflow")
SEED=2152435495

TOTAL=$(( ${#DATASETS[@]} * ${#METHODS[@]} ))
if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((TOTAL-1))). Exiting."
  exit 0
fi

DIDX=$(( SLURM_ARRAY_TASK_ID / ${#METHODS[@]} ))
MIDX=$(( SLURM_ARRAY_TASK_ID % ${#METHODS[@]} ))

DATASET=${DATASETS[$DIDX]}
ZCP_METHOD=${METHODS[$MIDX]}

echo "Precompute and verify op ZCPs: dataset=${DATASET}, method=${ZCP_METHOD}, seed=${SEED}"

python -u naslib/optimizers/oneshot/gsparsity/zc_score_op_time_saver.py \
  --dataset "${DATASET}" \
  --zcp_method "${ZCP_METHOD}" \
  --seed "${SEED}" \
  --out_base_dir naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_op_scoring_timefactor \
  --arch_scores_dir naslib/optimizers/oneshot/gsparsity/submission_scripts/nasbench201_zc_scoring_timefactor \
  --prune_topk 750