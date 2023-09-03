import cudf
import pandas as pd
from cuml.neighbors import KNeighborsClassifier
from cupy import asnumpy
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from cuml.preprocessing import StandardScaler as cuStandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from colorama import Fore
from sklearn.svm import SVC

def dnn(df_label, df_type):
    # Splitting the datasets
    X_label = df_label.drop(columns=['Attack_label'])
    y_label = df_label['Attack_label']

    X_type = df_type.drop(columns=['Attack_type'])
    y_type = df_type['Attack_type']

    # Splitting data into train and test sets
    X_label_train, X_label_test, y_label_train, y_label_test = train_test_split(X_label, y_label, test_size=0.3, random_state=42)
    X_type_train, X_type_test, y_type_train, y_type_test = train_test_split(X_type, y_type, test_size=0.3, random_state=42)

    # Scaling the data
    scaler_label = StandardScaler().fit(X_label_train)
    X_label_train_scaled = scaler_label.transform(X_label_train)
    X_label_test_scaled = scaler_label.transform(X_label_test)

    scaler_type = StandardScaler().fit(X_type_train)
    X_type_train_scaled = scaler_type.transform(X_type_train)
    X_type_test_scaled = scaler_type.transform(X_type_test)

    print(Fore.GREEN + "DNN for solving Binary Classification - (Attack or Not)" + Fore.RESET)

    # Building DNN for binary classification (Attack_label)
    model_label = tf.keras.Sequential([
        tf.keras.layers.Dense(21, activation='relu', input_shape=(X_label_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_label.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_label = model_label.fit(X_label_train_scaled, y_label_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    df_loss = pd.DataFrame({
        'Epoch': range(1, len(history_label.history['loss']) + 1),
        'Training_Loss': history_label.history['loss'],
        'Validation_Loss': history_label.history['val_loss']
    })

    # Create a Seaborn lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Epoch', y='Training_Loss', data=df_loss, marker='o', label='Training Loss')
    sns.lineplot(x='Epoch', y='Validation_Loss', data=df_loss, marker='s', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve for DNN - Binary Classification')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(Fore.GREEN + "DNN for solving Multiple Classification - (Which Attack)" + Fore.RESET)

    # Building DNN for multi-class classification (Attack_type)
    model_type = tf.keras.Sequential([
        tf.keras.layers.Dense(21, activation='relu', input_shape=(X_type_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(y_type.unique()), activation='softmax')
    ])

    model_type.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_type = model_type.fit(X_type_train_scaled, y_type_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    df_loss_type = pd.DataFrame({
        'Epoch': range(1, len(history_type.history['loss']) + 1),
        'Training_Loss': history_type.history['loss'],
        'Validation_Loss': history_type.history['val_loss']
    })

    # Create a Seaborn lineplot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Epoch', y='Training_Loss', data=df_loss_type, marker='o', label='Training Loss')
    sns.lineplot(x='Epoch', y='Validation_Loss', data=df_loss_type, marker='s', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve for DNN - Binary Classification')
    plt.legend()
    plt.grid(True)
    plt.show()


    y_label_pred = (model_label.predict(X_label_test_scaled) > 0.5).astype("int32")
    accuracy_label = accuracy_score(y_label_test, y_label_pred)
    report_label = classification_report(y_label_test, y_label_pred)

    print(f"Accuracy: {accuracy_label}")
    print(report_label)

    # Create confusion matrix
    cm_label = confusion_matrix(y_label_test, y_label_pred)

    # Plot heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_label, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for Binary Classification - DNN')
    plt.show()

    # Predict on the test set
    y_type_pred = model_type.predict(X_type_test_scaled)
    y_type_pred_classes = y_type_pred.argmax(axis=1)

    # Compute evaluation metrics
    accuracy_type = accuracy_score(y_type_test, y_type_pred_classes)
    report_type = classification_report(y_type_test, y_type_pred_classes)

    print(f"Accuracy: {accuracy_type}")
    print(report_type)

        # Create confusion matrix
    cm_type = confusion_matrix(y_type_test, y_type_pred_classes)

    # Plot heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(cm_type, annot=True, fmt='g',cmap='Blues', norm=colors.LogNorm())
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for Multi-Class Classification - DNN')
    plt.show()

def alternative_methods(df_label, df_type):
    # Preparing the data for binary classification
    X_label = df_label.drop(columns=['Attack_label'])
    y_label = df_label['Attack_label']

    # Preparing the data for multiple classification
    df_type_cudf = cudf.DataFrame.from_pandas(df_type)
    X_label_knn = df_type_cudf.drop(columns=['Attack_type'])
    y_label_knn = df_type_cudf['Attack_type']

    # Splitting data into train and test sets for binary classification
    X_label_train, X_label_test, y_label_train, y_label_test = train_test_split(X_label, y_label, test_size=0.3, random_state=42)

    # Splitting data into train and test sets for multiplke classification
    X_label_train_knn, X_label_test_knn, y_label_train_knn, y_label_test_knn = train_test_split(X_label_knn, y_label_knn, test_size=0.3, random_state=42)

    # Scaling the data
    scaler_label = StandardScaler().fit(X_label_train)
    X_label_train_scaled = scaler_label.transform(X_label_train)
    X_label_test_scaled = scaler_label.transform(X_label_test)

    # Scaling the data
    scaler_label = cuStandardScaler().fit(X_label_train_knn)
    X_label_train_knn_scaled = scaler_label.transform(X_label_train_knn)
    X_label_test_knn_scaled = scaler_label.transform(X_label_test_knn)

    # Training SVM for binary classification
    svm_label = SVC(kernel='linear', probability=True)
    svm_label.fit(X_label_train_scaled, y_label_train)

    # Training KNN for binary classification
    accuracies_knn = []
    f1_scores_knn = []
    neighbors = list(range(3,30))
    y_label_pred_knn_vector =[]

    for k in tqdm(neighbors, desc="Processing KNN", unit="neighbors"):
        knn_label = KNeighborsClassifier(n_neighbors=k)
        knn_label.fit(X_label_train_knn_scaled, y_label_train_knn)
        y_label_pred_knn_vector.append(knn_label.predict(X_label_test_knn_scaled))
        accuracies_knn.append(accuracy_score( asnumpy(y_label_test_knn), asnumpy(y_label_pred_knn_vector[-1])))
        f1_scores_knn.append(f1_score(asnumpy(y_label_test_knn), asnumpy(y_label_pred_knn_vector[-1]), average='weighted'))

        
    # Plotting the accuracies for different k-values
    plt.figure(figsize=(10,6))
    sns.lineplot(x=neighbors, y=accuracies_knn, marker='o', linestyle='-')
    plt.title('KNN Accuracy for different k values')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

    # Plotting the f1-scores for different k-values
    plt.figure(figsize=(10,6))
    sns.lineplot(x=neighbors, y=f1_scores_knn, marker='o', linestyle='-')
    plt.title('KNN F1-Score for different k values')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('F1-Score')
    plt.grid(True)
    plt.show()


    # Find optimal k (highest accuracy) without overfitting
    optimal_k_acc = neighbors[accuracies_knn.index(max(accuracies_knn))]
    optimal_k_f1 = neighbors[f1_scores_knn.index(max(f1_scores_knn))]
    print(f"The optimal number of neighbors is {optimal_k_acc} based on accuracies")
    print(f"The optimal number of neighbors is {optimal_k_f1} based on F1-Score")

    # Using optimal k for final evaluation
    y_label_pred_knn = y_label_pred_knn_vector[optimal_k_acc]

    # Metrics for the optimal k
    precision_kf1 = precision_score(asnumpy(y_label_test_knn), asnumpy(y_label_pred_knn))
    recall_kf1= recall_score(asnumpy(y_label_test_knn), asnumpy(y_label_pred_knn))
    f1_kf1 = f1_score(asnumpy(y_label_test_knn), asnumpy(y_label_pred_knn), average='weighted')

    print(f"Precision: {precision_kf1}")
    print(f"Recall: {recall_kf1}")
    print(f"F1-score: {f1_kf1}")


    # Confusion matrix for knn
    cm_knn = confusion_matrix(y_label_test, y_label_pred_knn)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm_knn, annot=True, fmt='d',cmap='Blues', norm=colors.LogNorm())
    plt.title("Confusion Matrix - KNN")
    plt.show()

    # Predicting on the test set
    y_label_pred_svm = svm_label.predict(X_label_test_scaled)
    accuracy_label_svm = accuracy_score(y_label_test, y_label_pred_svm)

    print(accuracy_label_svm)

    cm_svm = confusion_matrix(y_label_test, y_label_pred_svm)
    # Plot heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_svm, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for Binary Classification - SVM')
    plt.show()