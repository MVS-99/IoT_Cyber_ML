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
        if os.listdir("./CSV"):
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
        import modeling
        import pandas as pd
        # Initiate EDA analysis
        print_message(Fore.BLUE, "------------- RUNNING: EDA (Exploratory Data Analysis) -------------")
        eda.run_eda()
        print_message(Fore.BLUE, "------------- RUNNING: FEATURE SELECTION -------------")
        # Select features
        df_train = pd.read_csv('CSV/preprocessed_DNN.csv', low_memory=False)
        df_features_type = fselection.select_features(df_train, 'Attack_type', 'Attack_label',21)
        df_features_label = fselection.select_features(df_train, 'Attack_label', 'Attack_type',21)
        df_features_type.to_csv('CSV/type_dataframe_DNN.csv', encoding='utf-8', index=False)
        df_features_label.to_csv('CSV/label_dataframe_DNN.csv', encoding='utf-8', index=False)
        
        print_message(Fore.BLUE, "------------- RUNNING: MODELING -------------")
        modeling.dnn(df_features_label,df_features_type)


if __name__ == "__main__":
    main()