import os


def generate_slurm_script(optimizer, dataset, seed):
    script_content = f"""#!/bin/bash
#SBATCH --time=5:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=16
#SBATCH --job-name=wide_hpo_{optimizer}_{dataset}_{seed}
#SBATCH --output=naslib/optimizers/oneshot/gsparsity/results/{optimizer}/{dataset}/slurm/wide_hpo_{seed}_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruben.weber@students.uni-mannheim.de

python naslib/optimizers/oneshot/gsparsity/wide_hpo_search_space_configurator_loss_based.py \\
    --optimizer {optimizer} \\
    --search_space nasbench201 \\
    --dataset {dataset} \\
    --seed {seed} \\
    --out_dir naslib/optimizers/oneshot/gsparsity/results/{optimizer}/{dataset}/wide_hpo \\
"""
    return script_content


def main():
    optimizers = [
        "gsparsity",
        "zcp_gsparsity",
        "inverted_bananas_gsparsity",
        "inverted_bananas_zcp_gsparsity",
    ]
    datasets = ["cifar10", "cifar100", "imagenet"]
    seed = 2152435495

    os.makedirs("output", exist_ok=True)

    for optimizer in optimizers:
        for dataset in datasets:
            script = generate_slurm_script(optimizer, dataset, seed)
            output_dir = f"naslib/optimizers/oneshot/gsparsity/submission_scripts/{optimizer}/{dataset}"
            os.makedirs(output_dir, exist_ok=True)
            script_filename = f"{output_dir}/wide_hpo_{seed}.sh"
            with open(script_filename, "w") as f:
                f.write(script)


if __name__ == "__main__":
    main()
