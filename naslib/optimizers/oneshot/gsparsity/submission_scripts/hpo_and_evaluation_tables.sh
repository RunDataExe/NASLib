#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpuonly
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=table_eval
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/slurm/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

set -e -o pipefail

module --quiet --force purge
module load compiler/intel/2025.1_llvm
module load numlib/mkl/2025.1

# Conda init
if [[ -z "${CONDA_BASE:-}" ]]; then
  for cand in \
    "/hkfs/work/workspace/scratch/ma_ruweber-gs_nas_extension/miniconda3" \
    "$HOME/miniconda3" "$HOME/.miniconda3" "/hkfs/home/$USER/miniconda3"
  do
    [[ -f "$cand/etc/profile.d/conda.sh" ]] && CONDA_BASE="$cand" && break
  done
fi
if [[ -z "${CONDA_BASE:-}" ]] || [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "[Error] Could not find conda.sh. Set CONDA_BASE." >&2
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate 38_gs_nas

cd "$SLURM_SUBMIT_DIR"

mkdir -p naslib/optimizers/oneshot/gsparsity/table_scripts/tables/{sva,qva}

echo "[1] Final HP setting tables (SVA & QVA)"
python naslib/optimizers/oneshot/gsparsity/table_scripts/final_hp_setting_table_sva_shifted.py
python naslib/optimizers/oneshot/gsparsity/table_scripts/final_hp_setting_table_shifted.py

echo "[2] Hyperparameter importance (SVA)"
python naslib/optimizers/oneshot/gsparsity/table_scripts/hpo_importance.py \
  --db_dir naslib/optimizers/oneshot/gsparsity/results_wide_hpo/WHPO_Databases \
  --output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/sva/hyperparameter_importance_tables.xlsx

echo "[3] Hyperparameter importance (QVA)"
python naslib/optimizers/oneshot/gsparsity/table_scripts/hpo_importance.py \
  --db_dir naslib/optimizers/oneshot/gsparsity/results_wide_hpo_queried_val_acc/WHPO_Databases_filtered \
  --output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/qva/hyperparameter_importance_tables.xlsx

echo "[4] Benchmark deployment performance (SVA)"
python naslib/optimizers/oneshot/gsparsity/table_scripts/final_hp_benchmark_deployment_performance_table_both_shifted_scoring.py \
  --root_dir naslib/optimizers/oneshot/gsparsity/result_final_hp \
  --output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/sva/benchmark_deployment_performance.xlsx \
  --latex_output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/sva/benchmark_deployment_performance.txt

echo "[5] Benchmark deployment performance (QVA)"
python naslib/optimizers/oneshot/gsparsity/table_scripts/final_hp_benchmark_deployment_performance_table_both_shifted_scoring.py \
  --root_dir naslib/optimizers/oneshot/gsparsity/result_final_hp_queried_val_acc \
  --output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/qva/benchmark_deployment_performance.xlsx \
  --latex_output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/qva/benchmark_deployment_performance.txt

echo "[6] Gsparsity deployment performance (SVA)"
python naslib/optimizers/oneshot/gsparsity/table_scripts/final_hp_gsparsity_deployment_performance_table_both_shifted_scoring.py \
  --root_dir naslib/optimizers/oneshot/gsparsity/result_final_hp \
  --output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/sva/gsparsity_deployment_performance.xlsx \
  --latex_output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/sva/gsparsity_deployment_performance.txt

echo "[7] Fixed-time comparison (SVA)"
python naslib/optimizers/oneshot/gsparsity/table_scripts/fixed_time_final_hp_performance_stability_table_both_shifted_scoring.py \
  --root_dir naslib/optimizers/oneshot/gsparsity/result_final_hp \
  --output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/sva/fixed_time_comparison.xlsx \
  --latex_output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/sva/fixed_time_comparison.txt

echo "[8] Fixed-time comparison (QVA)"
python naslib/optimizers/oneshot/gsparsity/table_scripts/fixed_time_final_hp_performance_stability_table_both_shifted_scoring.py \
  --root_dir naslib/optimizers/oneshot/gsparsity/result_final_hp_queried_val_acc \
  --output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/qva/fixed_time_comparison.xlsx \
  --latex_output naslib/optimizers/oneshot/gsparsity/table_scripts/tables/qva/fixed_time_comparison.txt \
  --rs_extra_time 1324210

echo "All table generation finished."