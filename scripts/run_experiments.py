import os
import logging
from scripts.helper import *

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)


def run_experiments(data_dir, outputs_dir, reps=10000):

    logging.info("Running ASOS Experiments...")
    test_asos(data_dir, outputs_dir, reps=reps)

    try:
        logging.info("Running Microfinance Experiments...")
        test_microfinance(data_dir, outputs_dir, reps=reps)
    except FileNotFoundError as e:
        logging.info("Missing dataset: run dataset_download.sh and preprocess.py")
        logging.info(e)

    try:
        logging.info("Running Gaussian Experiments...")
        # test_gaussian(data_dir, outputs_dir, reps=1000, T=5 * 10**4)
        test_gaussian(data_dir, outputs_dir, reps=1000, T=50000)
    except FileNotFoundError as e:
        logging.info("Missing dataset: run dataset_download.sh and preprocess.py")
        logging.info(e)

    try:
        logging.info("Running Upworthy Experiments...")
        test_upworthy(data_dir, outputs_dir, reps=reps)
    except FileNotFoundError as e:
        logging.info("Missing dataset: run dataset_download.sh and preprocess.py")
        logging.info(e)

    try:
        logging.info("Running ASOS Group Experiments...")
        test_asos(data_dir, outputs_dir, reps=reps)
    except FileNotFoundError as e:
        logging.info("Missing dataset: run dataset_download.sh and preprocess.py")
        logging.info(e)

    try:
        logging.info("Running LLMbench Experiments...")
        test_llmbench(data_dir, outputs_dir, reps=100)
    except FileNotFoundError as e:
        logging.info("Missing dataset: run dataset_download.sh and preprocess.py")
        logging.info(e)


def main():
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, "../data")
    outputs_dir = os.path.join(dirname, "../outputs")
    run_experiments(data_dir, outputs_dir)


if __name__ == "__main__":
    main()
