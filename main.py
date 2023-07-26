## This is the main script for the analysis of the Edge IIoT dataset,
## is at your disposal for reproducibility

from preprocessing import *

if not 'colorama' in {pkg.key for pkg in pkg_resources.working_set}:
    subprocess.run(["conda", "install", "-c", "conda-forge", "colorama"])
from colorama import Fore

def print_message(color, message):
    border = '#' * len(message)
    print(color + border + Fore.RESET)
    print(color + message + Fore.RESET)
    print(color + border + Fore.RESET)

def welcome_message():
    control = False
    print_message(Fore.GREEN, "\nWelcome to the analysis of the IoT-Edge dataset.\n Let's run the prerequisites check, please be patient")
    
    print_message(Fore.BLUE, "------------- RUNNING: LIBRARY CHECK -------------")
    library_check()

    print_message(Fore.BLUE, "------------- RUNNING: KAGGLE API JSON CHECK -------------")
    control = kaggle_api_json()

    if (control == True):
        if os.path.exists("preprocessed_DNN.csv"):
            print_message(Fore.YELLOW, "WARNING: The preprocessed dataset has already been created, generation process will be bypassed.")

        else:
            print_message(Fore.BLUE, "------------- RUNNING: DOWNLOAD RAW CSV -------------")
            download_raw_csv()

            print_message(Fore.BLUE, "------------- RUNNING: PREPROCESSING -------------")
            preprocessing()

        print_message(Fore.BLUE, "------------- COMPLETED: ALL STEPS -------------")
        print_message(Fore.GREEN, "The preprocessed data is now ready for analysis.")
        return True
    else:
        print_message(Fore.RED, "ERROR: Please solve the required dependencies")
        return False


def main():
    # Initiate the user interaction with the data-set, preprocessing and prerequisites
    control_welcome = False
    control_welcome = welcome_message()
    if(control_welcome == True):
        import fselection
        import eda
        import pandas as pd
        # Initiate EDA analysis
        print_message(Fore.BLUE, "------------- RUNNING: EDA (Exploratory Data Analysis) -------------")
        eda.run_eda()
        print_message(Fore.BLUE, "------------- RUNNING: FEATURE SELECTION -------------")
        # Select features
        df = pd.read_csv('preprocessed_DNN.csv', low_memory=False)
        print_message(Fore.BLUE, "------------- RUNNING: OUTLIER DETECTION AND REMOVAL -------------")
        # Detect and remove outliers
        for num_features in range(22, 0, -1):
            # Perform feature selection
            df_features_type = fselection.select_features(df, 'Attack_type', 'Attack_label',num_features)
            df_features_label = fselection.select_features(df, 'Attack_label', 'Attack_type',num_features)
            # Perform outlier detection and removal using one-class SVM
            df_outliers_type = fselection.detect_and_remove_outliers_svm(df_features_type, nu=0.1, kernel='linear')
            df_outliers_label = fselection.detect_and_remove_outliers_svm(df_features_label, nu=0.1, kernel='linear')


if __name__ == "__main__":
    main()