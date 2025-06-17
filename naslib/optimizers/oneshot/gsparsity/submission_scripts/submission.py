import os
import subprocess
import argparse


def sbatch_all_sh_in_folder(folder_path):
    """
    Finds all .sh files in the given folder and submits them using sbatch.

    Args:
        folder_path (str): The path to the folder containing the .sh files.
    """
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"Error: Folder not found at {folder_path}")
        return
    except Exception as e:
        print(f"Error accessing folder {folder_path}: {e}")
        return

    sh_files_found = False
    for filename in files:
        if filename.endswith(".sh"):
            sh_files_found = True
            script_path = os.path.join(folder_path, filename)
            try:
                print(f"Submitting {script_path}...")
                result = subprocess.run(
                    ["sbatch", script_path], capture_output=True, text=True, check=True
                )
                print(f"Successfully submitted {filename}. Output:\n{result.stdout}")
                if result.stderr:
                    print(f"Stderr:\n{result.stderr}")
            except FileNotFoundError:
                print(
                    "Error: sbatch command not found. Make sure Slurm is installed and in your PATH."
                )
                return  # Stop if sbatch is not found
            except subprocess.CalledProcessError as e:
                print(f"Error submitting {filename}:")
                print(f"Return code: {e.returncode}")
                print(f"Output:\n{e.stdout}")
                print(f"Error output:\n{e.stderr}")
            except Exception as e:
                print(f"An unexpected error occurred while submitting {filename}: {e}")

    if not sh_files_found:
        print(f"No .sh files found in {folder_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit all .sh files in a specified folder using sbatch."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        help="The path to the folder containing the .sh files.",
        required=True,
    )
    args = parser.parse_args()

    target_folder = args.folder_path

    if not os.path.isdir(target_folder):
        print(f"Error: The provided path '{target_folder}' is not a valid directory.")
    else:
        print(f"Looking for .sh files in: {target_folder}")
        sbatch_all_sh_in_folder(target_folder)
