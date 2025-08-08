import os


def _script_header(optimizer, dataset, seed, zcp_method=None):
    job_suffix = f"_{zcp_method}" if zcp_method else ""
    zcp_out_suffix = f"_{zcp_method}" if zcp_method else ""
    return f"""#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=wide_hpo_{optimizer}_{dataset}_{seed}{job_suffix}
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/results/slurm/wide_hpo_{optimizer}_{dataset}_{seed}{zcp_out_suffix}_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

# Fail fast on errors; keep pipelines strict
set -eo pipefail

# Safe defaults: minimize thread contention across libs and processes
CPUS=${{SLURM_CPUS_PER_TASK:-32}}
THREADS=${{THREADS:-1}}  # override per job if needed

echo "[Slurm] CPUs-per-task=$CPUS -> per-proc threads=$THREADS"

# Limit BLAS/OpenMP/PyTorch intra-op threads to avoid oversubscription
export OMP_NUM_THREADS=${{OMP_NUM_THREADS:-$THREADS}}
export MKL_NUM_THREADS=${{MKL_NUM_THREADS:-$THREADS}}
export OPENBLAS_NUM_THREADS=${{OPENBLAS_NUM_THREADS:-$THREADS}}
export NUMEXPR_NUM_THREADS=${{NUMEXPR_NUM_THREADS:-$THREADS}}
export TORCH_NUM_THREADS=${{TORCH_NUM_THREADS:-$THREADS}}

# Reduce busy-waiting on shared CPUs
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
export MKL_DYNAMIC=FALSE
# Optional pinning (can hurt on shared nodes): export KMP_AFFINITY=granularity=fine,compact,1,0

# Unbuffered Python output for timely logs
export PYTHONUNBUFFERED=1

# Higher file descriptor limit (harmless if capped by the system)
ulimit -n ${{ULIMIT_NOFILE:-16384}} || true

"""


def generate_slurm_script(optimizer, dataset, seed, zcp_method):
    header = _script_header(optimizer, dataset, seed, zcp_method)

    base_cmd = [
        "python -u naslib/optimizers/oneshot/gsparsity/high_budget_wide_hpo_search_space_configurator_val_acc_based.py",
        f"--optimizer {optimizer}",
        "--search_space nasbench201",
        f"--dataset {dataset}",
        f"--seed {seed}",
        "--resume True",
        "--out_dir naslib/optimizers/oneshot/gsparsity/results",
    ]
    if zcp_method:
        base_cmd.insert(2, f"--zcp_method {zcp_method}")

    cmd_str = " \\\n    ".join(base_cmd) + "\n"
    return header + cmd_str


def main():
    optimizers = [
        "gsparsity",
        "zcp_gsparsity",
        "zcp-pre_gsparsity",
        "zcp-pre_zcp_gsparsity",
        # "inverted_bananas_gsparsity",
        # "inverted_bananas_zcp_gsparsity",
    ]
    datasets = ["cifar100", "ImageNet16-120"]
    zcp_method = ["jacov", "params", "synflow"]
    seed = 2152435495

    for optimizer in optimizers:
        for dataset in datasets:
            output_dir = f"naslib/optimizers/oneshot/gsparsity/submission_scripts/wide_hpo/{optimizer}"
            os.makedirs(output_dir, exist_ok=True)
            if "zcp_gsparsity" in optimizer:
                for zcp in zcp_method:
                    script_filename = f"{output_dir}/{dataset}_{zcp}_{seed}.sh"
                    with open(script_filename, "w") as f:
                        f.write(generate_slurm_script(optimizer, dataset, seed, zcp))
            else:
                script_filename = f"{output_dir}/{dataset}_{seed}.sh"
                with open(script_filename, "w") as f:
                    f.write(generate_slurm_script(optimizer, dataset, seed, None))


if __name__ == "__main__":
    main()
