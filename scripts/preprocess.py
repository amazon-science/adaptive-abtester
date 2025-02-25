import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import spacy
from tqdm import tqdm
from abtester.optimizers import *


def create_directory_if_not_exists(dir_path):
    """
    Creates a directory if it doesn't exist.
    Prints an error message if creation fails.
    """
    try:
        os.mkdir(dir_path)
    except OSError as error:
        print(error)


# --------------------------------------------------------------------------------
# Data Generation
# --------------------------------------------------------------------------------


def generate_gaussian_data(T, data_dir, outputs_dir):
    """
    Generates Gaussian data with different standard deviations for y1 and y0,
    saves them as CSV files, and creates corresponding output directories.
    """

    mu1, mu0 = 2, 1
    # sigmas = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    sigmas = [0.1, 1, 10]

    os.makedirs(os.path.join(data_dir, "gaussian"), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, "gaussian"), exist_ok=True)
    for sigma1 in sigmas:
        for sigma0 in sigmas:
            y1 = np.random.normal(mu1, sigma1, T)  # * 1000#/ 1000
            y0 = np.random.normal(mu0, sigma0, T)  # * 1000#/ 1000
            df = pd.DataFrame({"y1": y1, "y0": y0})

            dataset_name = f"indep_{sigma0}_{sigma1}.csv"
            df.to_csv(os.path.join(data_dir, "gaussian", dataset_name), index=True)

            # dataset_output_path = os.path.join(outputs_dir, 'gaussian', dataset_name)
            # create_directory_if_not_exists(dataset_output_path)


# --------------------------------------------------------------------------------
# Data Processing
# --------------------------------------------------------------------------------


def data_processing_asos(source_dir, data_dir):
    """
    Reads ASOS data, groups by metric_id, and saves outcomes for each group to CSV.
    """
    df_asos = pd.read_csv(source_dir).dropna()
    df_asos_metric = [df_table for _, df_table in df_asos.groupby("metric_id")]
    num_metrics = len(df_asos_metric)

    dirname = os.path.join(data_dir, "asos")
    os.makedirs(dirname, exist_ok=True)

    for i in range(num_metrics):
        df = pd.DataFrame(
            {"y1": df_asos_metric[i]["mean_t"], "y0": df_asos_metric[i]["mean_c"]}
        )
        filepath = os.path.join(dirname, f"metric_{i}.csv")
        df.to_csv(filepath, index=True)


def data_processing_llmbench(source_dir, data_dir, outputs_dir):
    """
    Reads data from torch files, extracts accuracy and confidence metrics
    for each model, and saves them in corresponding subdirectories.
    """
    datasets = sorted(os.listdir(source_dir))
    dataset_dirs = []

    for dataset in datasets:
        path = os.path.join(source_dir, dataset)
        if os.listdir(path):
            dataset_filename = os.listdir(path)[0]
            dataset_dir = os.path.join(path, dataset_filename)
            dataset_dirs.append((dataset_dir, dataset))
    dataset_dirs = sorted(dataset_dirs)

    relevant_dataset_dirs = []
    datasets_targets_probs = []
    for dataset_dir, dataset in tqdm(dataset_dirs):
        df = torch.load(dataset_dir)
        columns = df.keys()
        if "Target" in columns and "AuxiliaryPredictionProb" in columns:
            relevant_dataset_dirs.append(dataset_dir)
            datasets_targets_probs.append(
                (dataset, df["Target"], df["AuxiliaryPredictionProb"])
            )
    relevant_dataset_dirs = sorted(relevant_dataset_dirs)
    datasets_targets_probs = sorted(datasets_targets_probs)

    datasets_stats = {}
    for ds_name, targets, probs in datasets_targets_probs:
        data_size, num_models, num_classes = np.array(probs).shape
        datasets_stats[ds_name] = {
            "data_size": data_size,
            "num_models": num_models,
            "num_classes": num_classes,
        }
    print(datasets_stats)

    model0_confidences, model1_confidences = {}, {}
    model0_max_confidences, model1_max_confidences = {}, {}
    model0_accuracies, model1_accuracies = {}, {}

    for ds_name, targets, probs in datasets_targets_probs:
        num_datapoints = datasets_stats[ds_name]["data_size"]

        model0_confidences[ds_name] = [
            probs[i][0][targets[i]] for i in range(num_datapoints)
        ]
        model1_confidences[ds_name] = [
            probs[i][1][targets[i]] for i in range(num_datapoints)
        ]
        model0_max_confidences[ds_name] = [
            np.max(probs[i][0]) for i in range(num_datapoints)
        ]
        model1_max_confidences[ds_name] = [
            np.max(probs[i][1]) for i in range(num_datapoints)
        ]
        model0_accuracies[ds_name] = [
            int(np.argmax(probs[i][0]) == targets[i]) for i in range(num_datapoints)
        ]
        model1_accuracies[ds_name] = [
            int(np.argmax(probs[i][1]) == targets[i]) for i in range(num_datapoints)
        ]

    for ds_name, targets, probs in datasets_targets_probs:
        prefix_path = os.path.join(data_dir, "llmbench")
        dataset_path = os.path.join(prefix_path, ds_name)
        create_directory_if_not_exists(dataset_path)

        prefix_output_path = os.path.join(outputs_dir, "llmbench")
        dataset_output_path = os.path.join(prefix_output_path, ds_name)
        create_directory_if_not_exists(dataset_output_path)

    for ds_name, targets, probs in datasets_targets_probs:
        accuracy_data_path = os.path.join(data_dir, "llmbench", ds_name, "accuracies")
        confidence_data_path = os.path.join(
            data_dir, "llmbench", ds_name, "confidences"
        )
        max_confidence_data_path = os.path.join(
            data_dir, "llmbench", ds_name, "max_confidences"
        )

        create_directory_if_not_exists(accuracy_data_path)
        create_directory_if_not_exists(confidence_data_path)
        create_directory_if_not_exists(max_confidence_data_path)

        df_confidences = pd.DataFrame(
            {"y1": model1_confidences[ds_name], "y0": model0_confidences[ds_name]}
        ).dropna()
        df_confidences.to_csv(
            os.path.join(confidence_data_path, "confidences.csv"), index=True
        )

        df_max_confidences = pd.DataFrame(
            {
                "y1": model1_max_confidences[ds_name],
                "y0": model0_max_confidences[ds_name],
            }
        ).dropna()
        df_max_confidences.to_csv(
            os.path.join(max_confidence_data_path, "max_confidences.csv"), index=True
        )

        df_accuracies = pd.DataFrame(
            {"y1": model1_accuracies[ds_name], "y0": model0_accuracies[ds_name]}
        ).dropna()
        df_accuracies.to_csv(
            os.path.join(accuracy_data_path, "accuracies.csv"), index=True
        )


if __name__ == "__main__":

    RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data/raw_data")
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

    try:
        data_processing_asos(
            source_dir=os.path.join(
                RAW_DATA_DIR, "asos_digital_experiments_dataset.csv"
            ),
            data_dir=DATA_DIR,
        )
    except FileNotFoundError as e:
        print("Missing dataset: run scripts/dataset_download.sh")
        print(e)

    try:
        data_processing_llmbench(
            source_dir=DATA_DIR + "/bigbench", outputs_dir=DATA_DIR, data_dir=DATA_DIR
        )
    except FileNotFoundError as e:
        print("Missing dataset: run scripts/dataset_download.sh")
        print(e)

    generate_gaussian_data(T=5 * 10**4, data_dir=DATA_DIR, outputs_dir=DATA_DIR)
