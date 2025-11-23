import os
import subprocess
import argparse


def sbatch_all_sh_in_folder(folder_path, recursive=False):
    """
    Finds .sh files in the given folder (and optionally subfolders) and submits them using sbatch.

    Args:
        folder_path (str): The path to the folder containing the .sh files.
        recursive (bool): Whether to include subdirectories.
    """
    try:
        if not recursive:
            candidates = [(folder_path, os.listdir(folder_path))]
        else:
            candidates = []
            for root, _, files in os.walk(folder_path):
                candidates.append((root, files))
    except FileNotFoundError:
        print(f"Error: Folder not found at {folder_path}")
        return
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")
        return

    sh_files_found = False
    for root, files in candidates:
        for filename in sorted(files):
            if filename.endswith(".sh"):
                sh_files_found = True
                script_path = os.path.join(root, filename)
                try:
                    print(f"Submitting {script_path}...")
                    result = subprocess.run(
                        ["sbatch", script_path],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    print(
                        f"Successfully submitted {filename}. Output:\n{result.stdout}"
                    )
                    if result.stderr:
                        print(f"Stderr:\n{result.stderr}")
                except FileNotFoundError:
                    print("Error: sbatch command not found. Slurm not in PATH.")
                    return
                except subprocess.CalledProcessError as e:
                    print(f"Error submitting {filename}:")
                    print(f"Return code: {e.returncode}")
                    print(f"Output:\n{e.stdout}")
                    print(f"Error output:\n{e.stderr}")
                except Exception as e:
                    print(f"Unexpected error while submitting {filename}: {e}")

    if not sh_files_found:
        scope = "recursively" if recursive else ""
        print(f"No .sh files found {scope} in {folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit .sh files in a specified folder using sbatch."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        help="Path to the folder containing the .sh files.",
        required=True,
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include .sh files in all subdirectories.",
    )
    args = parser.parse_args()

    target_folder = args.folder_path

    if not os.path.isdir(target_folder):
        print(f"Error: The provided path '{target_folder}' is not a valid directory.")
    else:
        mode = "recursively" if args.recursive else "in folder"
        print(f"Looking for .sh files {mode}: {target_folder}")
        sbatch_all_sh_in_folder(target_folder, recursive=args.recursive)
