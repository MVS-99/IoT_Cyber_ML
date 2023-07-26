## This is the main script for the analysis of the Edge IIoT dataset,
## is at your disposal for reproducibility

from preprocessing import *
from fselection import *

import eda

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from colorama import Fore, init

init()

def print_message(color, message):
    border = '#' * len(message)
    print(color + border + Fore.RESET)
    print(color + message + Fore.RESET)
    print(color + border + Fore.RESET)

def welcome_message():
    print_message(Fore.GREEN, "\nWelcome to the analysis of the IoT-Edge dataset.\n Let's run the prerequisites check, please be patient")
    
    print_message(Fore.BLUE, "------------- RUNNING: LIBRARY CHECK -------------")
    library_check()

    print_message(Fore.BLUE, "------------- RUNNING: KAGGLE API JSON CHECK -------------")
    kaggle_api_json()

    if os.path.exists("preprocessed_DNN.csv"):
        print_message(Fore.YELLOW, "WARNING: The preprocessed dataset has already been created, generation process will be bypassed.")

    else:
        print_message(Fore.BLUE, "------------- RUNNING: DOWNLOAD RAW CSV -------------")
        download_raw_csv()

        print_message(Fore.BLUE, "------------- RUNNING: PREPROCESSING -------------")
        preprocessing()

    print_message(Fore.BLUE, "------------- COMPLETED: ALL STEPS -------------")
    print_message(Fore.GREEN, "The preprocessed data is now ready for analysis.")


def main():
    # Initiate the user interaction with the data-set, preprocessing and prerequisites
    welcome_message()
    # Initiate EDA analysis
    print_message(Fore.BLUE, "------------- RUNNING: EDA () ANALYSIS -------------")
    eda.run_eda()
    print_message(Fore.BLUE, "------------- RUNNING: FEATURE SELECTION -------------")
    # Select features
    df = select_features(df, 'Attack_type', 10)
    print_message(Fore.BLUE, "------------- RUNNING: OUTLIER DETECTION AND REMOVAL -------------")
    # Detect and remove outliers
    df = detect_and_remove_outliers(df, 20, 0.1)


if __name__ == "__main__":
    main()