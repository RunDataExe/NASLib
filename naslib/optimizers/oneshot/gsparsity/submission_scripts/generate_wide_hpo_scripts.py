import os


def generate_slurm_script(optimizer, dataset, seed, zcp_method):
    if zcp_method:
        script_content = f"""#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=wide_hpo_{optimizer}_{dataset}_{seed}_{zcp_method}
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/results/slurm/wide_hpo_{optimizer}_{dataset}_{seed}_{zcp_method}_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/high_budget_wide_hpo_search_space_configurator_val_acc_based.py \\
    --optimizer {optimizer} \\
    --search_space nasbench201 \\
    --dataset {dataset} \\
    --seed {seed} \\
    --resume True \\
    --zcp_method {zcp_method} \\
    --out_dir naslib/optimizers/oneshot/gsparsity/results \\
"""
    else:
        script_content = f"""#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=wide_hpo_{optimizer}_{dataset}_{seed}
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/results/slurm/wide_hpo_{optimizer}_{dataset}_{seed}_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/high_budget_wide_hpo_search_space_configurator_val_acc_based.py \\
    --optimizer {optimizer} \\
    --search_space nasbench201 \\
    --dataset {dataset} \\
    --seed {seed} \\
    --resume True \\
    --out_dir naslib/optimizers/oneshot/gsparsity/results \\
"""
    return script_content


def main():
    optimizers = [
        "gsparsity",
        "zcp_gsparsity",
        "zcp-pre_gsparsity",
        "zcp-pre_zcp_gsparsity",
        # "inverted_bananas_gsparsity",
        # "inverted_bananas_zcp_gsparsity",
    ]
    # datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    datasets = ["cifar100", "ImageNet16-120"]

    zcp_method = ["jacov", "params", "synflow"]

    seed = 2152435495

    os.makedirs("output", exist_ok=True)

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
