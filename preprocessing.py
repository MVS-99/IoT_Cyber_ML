import os
import subprocess
import pkg_resources


def library_check():
    # Check if the kaggle package is installed
    if 'kaggle' in {pkg.key for pkg in pkg_resources.working_set}:
        print("The Kaggle package is already installed.")
    else:
        print("The Kaggle package is not installed. Initiating installation...")
        # Install the kaggle package
        subprocess.run(["conda", "install", "-c", "conda-forge", "kaggle"])

    # Check if the kaggle package is installed
    if 'matplotlib' in {pkg.key for pkg in pkg_resources.working_set}:
        print("The Matplotlib package is already installed.")
    else:
        print("The Matplotlib package is not installed. Initiating installation...")
        # Install the kaggle package
        subprocess.run(["conda", "install", "-c", "conda-forge", "matplotlib"])
        
    # Check if the kaggle package is installed
    if 'seaborn' in {pkg.key for pkg in pkg_resources.working_set}:
        print("The Seaborn package is already installed.")
    else:
        print("The Seaborn package is not installed. Initiating installation...")
        # Install the kaggle package
        subprocess.run(["conda", "install", "-c", "conda-forge", "seaborn"])

    # Check if the kaggle package is installed
    if 'pandas' in {pkg.key for pkg in pkg_resources.working_set}:
        print("The Pandas package is already installed.")
    else:
        print("The Pandas package is not installed. Initiating installation...")
        # Install the kaggle package
        subprocess.run(["conda", "install", "-c", "conda-forge", "pandas"])

    # Check if the kaggle package is installed
    if 'numpy' in {pkg.key for pkg in pkg_resources.working_set}:
        print("The Numpy package is already installed.")
    else:
        print("The Numpy package is not installed. Initiating installation...")
        # Install the kaggle package
        subprocess.run(["conda", "install", "-c", "conda-forge", "numpy"])

    # Check if the kaggle package is installed
    if 'scikit-learn' in {pkg.key for pkg in pkg_resources.working_set}:
        print("The Scikit-Learn package is already installed.")
    else:
        print("The Scikit-Learn package is not installed. Initiating installation...")
        # Install the kaggle package
        subprocess.run(["conda", "install", "-c", "conda-forge", "scikit-learn"])

    if 'tensorflow' in {pkg.key for pkg in pkg_resources.working_set}:
        print("The Tensorflow package is already installed.")
    else:
        print("The Tensorflow package is not installed. Initiating installation...")
        # Install the kaggle package
        subprocess.run(["conda", "install", "-c", "conda-forge", "tensorflow"])

def kaggle_api_json():
    # Define the path to the kaggle.json file
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")

    # Check if the kaggle.json file exists
    if os.path.exists(kaggle_json_path):
        print("The API JSON file is found.")
        # Set appropriate permissions for kaggle.json
        subprocess.run(["chmod", "600", kaggle_json_path])
        return True
    else:
        print("The kaggle.json file is not found in the directory.")
        print("Please read the instructions on how to obtain it from the following link: https://www.kaggle.com/docs/api")
        return False

def download_raw_csv():
    # Download the dataset
    subprocess.run(["kaggle", "datasets", "download", "-d", "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot", "-f", "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"])

    # Unzip the downloaded dataset
    subprocess.run(["unzip", "DNN-EdgeIIoT-dataset.csv.zip", "-d", "CSV"])

    # Remove the zip file
    os.remove("DNN-EdgeIIoT-dataset.csv.zip")

def preprocessing():
    # OBJECTIVE: Dropping data (Columns, duplicated rows, NAN, Null..)
    import pandas as pd
    import numpy as np
    
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split

    # Reading the Datasets' CSV file to a Pandas DataFrame
    df = pd.read_csv('CSV/DNN-EdgeIIoT-dataset.csv', low_memory=False)
    
    # Using stratified sampling based on the 'attack_label' column
    #df_train, df_sliced = train_test_split(df, test_size=1/3, stratify=df['Attack_label'], random_state=42)

  # Columns to drop
    drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4", 
                    "http.file_data","http.request.full_uri","icmp.transmit_timestamp",
                    "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",
                    "tcp.dstport", "udp.port", "mqtt.msg"]

    # Dropping unnecessary columns
    #df_sliced.drop(drop_columns, axis=1, inplace=True)
    #df_train.drop(drop_columns, axis=1, inplace=True)
    df.drop(drop_columns, axis=1, inplace=True)

    # Dropping NaN values
    #df_sliced.dropna(axis=0, how='any', inplace=True)
    #df_train.dropna(axis=0, how='any', inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    # Dropping duplicate rows
    #df_sliced.drop_duplicates(subset=None, keep="first", inplace=True)
    #df_train.drop_duplicates(subset=None, keep="first", inplace=True)
    df.drop_duplicates(subset=None, keep="first", inplace=True)

    # Shuffling the dataframe
    #df_sliced = shuffle(df_sliced)
    #df_train = shuffle(df_train)
    df = shuffle(df)

    # OBJECTIVE: Categorical data encoding (Dummy Encoding)
    def encode_text_dummy(df, name):

        dummies = pd.get_dummies(df[name])

        for x in dummies.columns:

            dummy_name = f"{name}-{x}"

            df[dummy_name] = dummies[x]

        df.drop(name, axis=1, inplace=True)

        # Encoding categorical features
    categorical_features = ['http.request.method', 'http.referer', 'http.request.version', 'dns.qry.name.len', 
                            'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic']
    
    #for feature in categorical_features:
        #if feature in df_sliced.columns:
            #encode_text_dummy(df_sliced, feature)
    
    #for feature in categorical_features:
        #if feature in df_train.columns:
            #encode_text_dummy(df_train, feature)


    for feature in categorical_features:
        if feature in df.columns:
            encode_text_dummy(df, feature)

    #df_sliced.to_csv('test_preprocessed_DNN.csv', encoding='utf-8', index=False)
    #df_train.to_csv('train_preprocessed_DNN.csv', encoding='utf-8', index=False)
    df.to_csv('CSV/preprocessed_DNN.csv', encoding='utf-8', index=False)