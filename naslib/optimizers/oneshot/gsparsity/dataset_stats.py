import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from collections import Counter
import numpy as np
import os

# Attempt to import ImageNet16 from naslib
try:
    from naslib.utils.DownsampledImageNet import ImageNet16
except ImportError:
    print("Warning: Could not import ImageNet16 from naslib.utils.DownsampledImageNet.")
    print(
        "Please ensure NASLib is installed and in your PYTHONPATH, or adjust the import path."
    )
    print("ImageNet16-120 distribution will not be calculated.")
    ImageNet16 = None


def print_class_distribution(dataset_name, labels, split_name):
    """Prints the class distribution for a given dataset split."""
    print(f"\nClass distribution for {dataset_name} ({split_name}):")
    if not labels:
        print("No labels found or dataset could not be loaded.")
        return

    # Convert to list if it's a tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()

    if not labels:  # Check again after potential tolist() if it resulted in empty
        print(
            f"Labels list is empty for {dataset_name} ({split_name}). Cannot calculate distribution."
        )
        return

    counts = Counter(labels)
    for class_id in sorted(counts.keys()):
        print(f"Class {class_id}: {counts[class_id]} samples")
    print(f"Total samples in {split_name}: {len(labels)}")
    if len(counts) > 0:
        print(f"Number of classes: {len(counts)}")


def get_imagenet16_120_distribution(data_root="./data"):
    """Loads ImageNet16-120 and prints its class distribution."""
    if ImageNet16 is None:
        print(
            "ImageNet16 class not available. Skipping ImageNet16-120 distribution calculation."
        )
        return

    dataset_name_fs = "ImageNet16-120"  # filesystem name
    dataset_print_name = "ImageNet16-120"  # name for printing

    abs_data_root = os.path.abspath(data_root)
    data_folder = os.path.join(abs_data_root, dataset_name_fs)
    print(f"Attempting to load {dataset_print_name} from {data_folder}...")

    # if not os.path.exists(data_folder):
    #     print(f"Error: Data folder {data_folder} not found for {dataset_print_name}.")
    #     print(
    #         f"Please ensure {dataset_print_name} is downloaded and placed in the correct directory structure: {data_folder}"
    #     )
    #     return

    try:
        train_data = ImageNet16(
            root=data_folder,
            train=True,
            transform=None,  # No transform needed for labels
            use_num_of_class_only=120,
        )
        if hasattr(train_data, "targets"):
            labels_train = train_data.targets
        else:
            print(
                f"Warning: '.targets' attribute not found for {dataset_print_name} train_data. Iterating to get labels."
            )
            labels_train = [label for _, label in train_data]
        print_class_distribution(dataset_print_name, labels_train, "Train")

        test_data = ImageNet16(
            root=data_folder,
            train=False,
            transform=None,  # No transform needed for labels
            use_num_of_class_only=120,
        )
        if hasattr(test_data, "targets"):
            labels_test = test_data.targets
        else:
            print(
                f"Warning: '.targets' attribute not found for {dataset_print_name} test_data. Iterating to get labels."
            )
            labels_test = [label for _, label in test_data]
        print_class_distribution(dataset_print_name, labels_test, "Test")

    except Exception as e:
        print(f"Could not load or process {dataset_print_name}: {e}")
        print(
            f"Ensure the dataset is correctly placed in {data_folder} and the ImageNet16 class is working."
        )


if __name__ == "__main__":
    # Define the root directory for datasets.
    # It defaults to './data' relative to where the script is run,
    # or uses the NASLIB_DATA environment variable if set.
    DEFAULT_DATA_DIR = "./data"
    DATA_ROOT_DIR = os.getenv("NASLIB_DATA", DEFAULT_DATA_DIR)

    print(f"Using data root directory: {os.path.abspath(DATA_ROOT_DIR)}")
    print(
        "Ensure this path is correct for your dataset locations, especially for ImageNet16-120."
    )

    get_imagenet16_120_distribution(data_root=DATA_ROOT_DIR)

    print("\n--- Script Finished ---")
