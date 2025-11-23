#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpuonly
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=plot_eval
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/slurm/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

set -e -o pipefail

module --quiet --force purge
module load compiler/intel/2025.1_llvm
module load numlib/mkl/2025.1

# Find and init conda
if [[ -z "${CONDA_BASE:-}" ]]; then
  for cand in \
    "/hkfs/work/workspace/scratch/ma_ruweber-gs_nas_extension/miniconda3" \
    "$HOME/miniconda3" "$HOME/.miniconda3" "/hkfs/home/$USER/miniconda3"
  do
    [[ -f "$cand/etc/profile.d/conda.sh" ]] && CONDA_BASE="$cand" && break
  done
fi
if [[ -z "${CONDA_BASE:-}" ]] || [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "[Error] Could not find conda.sh. Set CONDA_BASE to your Miniconda path." >&2
  exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate 38_gs_nas

cd "$SLURM_SUBMIT_DIR"

declare -A ROOT_TO_OUT=(
  ["naslib/optimizers/oneshot/gsparsity/result_final_hp"]="naslib/optimizers/oneshot/gsparsity/plotting_scripts/sva_evaluation"
  ["naslib/optimizers/oneshot/gsparsity/result_final_hp_queried_val_acc"]="naslib/optimizers/oneshot/gsparsity/plotting_scripts/qva_evaluation"
)

for ROOT in "${!ROOT_TO_OUT[@]}"; do
  OUT="${ROOT_TO_OUT[$ROOT]}"
  mkdir -p "$OUT"

  # Select per-root T markers (SVA vs QVA)
  if [[ "$ROOT" == "naslib/optimizers/oneshot/gsparsity/result_final_hp" ]]; then
    T_MARKERS="cifar10=19432,cifar100=20808,ImageNet16-120=36931"
  else
    T_MARKERS="cifar10=18000,cifar100=20000,ImageNet16-120=43165"
  fi

  echo "Plotting (raw stability) root=$ROOT out=$OUT"
  python naslib/optimizers/oneshot/gsparsity/plotting_scripts/final_hp_setting_raw_stability_plotting_both_shifted.py \
    --root_dir "$ROOT" \
    --out_dir "$OUT" \
    --combine_plots \
    --xscale log \
    --t_markers "$T_MARKERS"

  echo "Plotting (incumbent performance) root=$ROOT out=$OUT"
  python naslib/optimizers/oneshot/gsparsity/plotting_scripts/final_hp_setting_incumbent_performance_plotting_both_shifted.py \
    --root_dir "$ROOT" \
    --out_dir "$OUT" \
    --combine_plots \
    --xscale log \
    --t_markers "$T_MARKERS"

  echo "Plotting (percentile distribution) root=$ROOT out=$OUT"
  python naslib/optimizers/oneshot/gsparsity/plotting_scripts/final_hp_setting_precentile_plotting_both.py \
    --root_dir "$ROOT" \
    --out_dir "$OUT" \
    --combine_plots \
    --t_markers "$T_MARKERS"
done

echo "All plotting finished."