#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=cpuonly
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=random_search
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/slurm/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

# Fail fast, but avoid -u (conflicts with conda activate hooks)
set -e -o pipefail

# Modules (CPU-only; CUDA not needed)
module --quiet --force purge
module load compiler/intel/2025.1_llvm
module load numlib/mkl/2025.1
# module load devel/cuda/12.4  # not on cpuonly

# Optional: you can drop the MKL module to rely on conda’s MKL only.

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

# Work from submit dir so relative paths resolve
cd "$SLURM_SUBMIT_DIR"

# Ensure log/output dirs exist
mkdir -p naslib/optimizers/oneshot/gsparsity/slurm
OUT_DIR="naslib/optimizers/oneshot/gsparsity/result_final_hp"

# Run all combinations sequentially
for d in cifar10 cifar100 ImageNet16-120; do
  for s in 1544457859 2092269736 3788705088; do
    echo "Running: dataset=$d, seed=$s"
    python naslib/optimizers/oneshot/gsparsity/configurator.py \
      --optimizer random_search \
      --search_space nasbench201 \
      --dataset "$d" \
      --seed "$s" \
      --out_dir "$OUT_DIR" \
      --search_epochs 300 \
      --eval_epochs 1
  done
done