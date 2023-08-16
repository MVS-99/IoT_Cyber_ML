import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_label_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model_label.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_label.fit(X_label_train_scaled, y_label_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    print(Fore.GREEN + "DNN for solving Multiple Classification - (Which Attack)" + Fore.RESET)

    # Building DNN for multi-class classification (Attack_type)
    model_type = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_type_train_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(y_type.unique()), activation='softmax')
    ])

    model_type.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_type.fit(X_type_train_scaled, y_type_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

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
    X_label_knn = df_type.drop(columns=['Attack_type'])
    y_label_knn = df_label['Attack_type']

    # Splitting data into train and test sets for binary classification
    X_label_train, X_label_test, y_label_train, y_label_test = train_test_split(X_label, y_label, test_size=0.3, random_state=42)

    # Splitting data into train and test sets for multiplke classification
    X_label_train_knn, X_label_test_knn, y_label_train_knn, y_label_test_knn = train_test_split(X_label_knn, y_label_knn, test_size=0.3, random_state=42)

    # Scaling the data
    scaler_label = StandardScaler().fit(X_label_train)
    X_label_train_scaled = scaler_label.transform(X_label_train)
    X_label_test_scaled = scaler_label.transform(X_label_test)

    # Scaling the data
    scaler_label = StandardScaler().fit(X_label_train_knn)
    X_label_train_knn_scaled = scaler_label.transform(X_label_train_knn)
    X_label_test_knn_scaled = scaler_label.transform(X_label_test_knn)

    # Training SVM for binary classification
    svm_label = SVC(kernel='linear', probability=True)
    svm_label.fit(X_label_train_scaled, y_label_train)

    # Training KNN for binary classification
    accuracies_knn = []
    f1_scores_knn = []
    neighbors = list(range(1,30))
    knn = []

    for k in neighbors:
        knn_label = KNeighborsClassifier(n_neighbors=k)
        knn_label.fit(X_label_train_knn_scaled, y_label_train_knn)
        y_label_pred_knn = knn_label.predict(X_label_test_knn_scaled)
        accuracies_knn.append(accuracy_score(y_label_test_knn, y_label_pred_knn))
        f1_scores_knn.append(f1_score(y_label_test, y_label_pred_knn))
        
    # Plotting the accuracies for different k-values
    plt.figure(figsize=(10,6))
    sns.lineplot(neighbors, accuracies_knn, marker='o', linestyle='-')
    plt.title('KNN Accuracy for different k values')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

    # Find optimal k (highest accuracy) without overfitting
    optimal_k_acc = neighbors[accuracies_knn.index(max(accuracies_knn))]
    optimal_k_f1 = neighbors[f1_scores_knn.index(max(f1_scores_knn))]
    print(f"The optimal number of neighbors is {optimal_k_acc} based on accuracies")
    print(f"The optimal number of neighbors is {optimal_k_f1} based on F1-Score")

    # Using optimal k for final evaluation
    knn_label = KNeighborsClassifier(n_neighbors=optimal_k_f1)
    knn_label.fit(X_label_train_scaled, y_label_train)
    y_label_pred_knn = knn_label.predict(X_label_test_scaled)

    # Metrics for the optimal k
    precision_kf1 = precision_score(y_label_test, y_label_pred_knn)
    recall_kf1= recall_score(y_label_test, y_label_pred_knn)
    f1_kf1 = f1_score(y_label_test, y_label_pred_knn)

    print(f"Precision: {precision_kf1}")
    print(f"Recall: {recall_kf1}")
    print(f"F1-score: {f1_kf1}")


    # Confusion matrix for knn
    cm = confusion_matrix(y_label_test, y_label_pred_knn)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, 
                xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.title("Confusion Matrix - KNN")
    plt.show()

    # Predicting on the test set
    y_label_pred_svm = svm_label.predict(X_label_test_scaled)
    accuracy_label_svm = accuracy_score(y_label_test, y_label_pred_svm)

    print(accuracy_label_svm)