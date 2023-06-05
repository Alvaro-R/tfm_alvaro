"""
Autor: Álvaro Román Gómez.

Tools to load data from different files and feed models.
"""

# IMPORTS
import os
import re
from typing import List, Optional
import pandas as pd
import glob

from regex import Match


def get_dataset_names(files: List[str]) -> List[str]:
    """
    Extract dataset names from a list of file names.

    Args:
        files (List[str]): A list of file names.

    Returns:
        List[str]: A list of dataset names extracted from the file names.
    """
    dataset_names: List[str] = []
    for file in files:
        file_match: Optional[Match[str]] = re.search(r"https//(.+?)/(.+)", file)
        if file_match is not None:
            dataset_names.append(file_match[1])
        else:
            print(f"File {file} does not match the expected pattern.")
    return dataset_names


def get_csv_files(path: str) -> list:
    """
    Retrieve a list of CSV file names from a given directory path.

    Args:
        path (str): The directory path to search for CSV files.

    Returns:
        List[str]: A list of CSV file names found in the directory.
    """
    files = glob.glob(os.path.join(path, "*.csv"))
    return files


def load_training_test_datasets(
    datasets_names: list, training_path: str, test_path: str
):
    """
    Load training and test datasets for multiple datasets.

    Args:
        datasets_names (List[str]): A list of dataset names.
        training_path (str): The directory path where the training datasets are located.
        test_path (str): The directory path where the test datasets are located.

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: A dictionary containing the loaded datasets,
        with the dataset names as keys and a nested dictionary as values.
        The nested dictionary contains the training and test data,
        with the keys 'X_train_{dataset_name}',
        'Y_train_{dataset_name}', 'X_test_{dataset_name}',
        and 'Y_test_{dataset_name}'.
    """
    # CREATE DICTIONARY TO SAVE DATASET TRAINING AND TEST DATA
    datasets = {}

    for dataset_name in datasets_names:
        # CREATE DICTIONARY TO SAVE DATASET TRAINING AND TEST DATA
        dataset = {}

        # LOAD TRAINING DATA
        X_train = pd.read_csv(
            os.path.join(training_path, f"{dataset_name}_training.csv")
        ).drop(columns=["activity"])
        Y_train = pd.read_csv(
            os.path.join(training_path, f"{dataset_name}_training.csv")
        )["activity"]

        # LOAD TEST DATA
        X_test = pd.read_csv(os.path.join(test_path, f"{dataset_name}_test.csv")).drop(
            columns=["activity"]
        )
        Y_test = pd.read_csv(os.path.join(test_path, f"{dataset_name}_test.csv"))[
            "activity"
        ]

        # SAVE DATA IN DICTIONARY
        dataset[f"X_train_{dataset_name}"] = X_train
        dataset[f"Y_train_{dataset_name}"] = Y_train
        dataset[f"X_test_{dataset_name}"] = X_test
        dataset[f"Y_test_{dataset_name}"] = Y_test

        # SAVE DATASET IN DICTIONARY
        datasets[dataset_name] = dataset

    return datasets
