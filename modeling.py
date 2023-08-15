import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore

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
    sns.heatmap(cm_type, annot=True, fmt='g',cmap='Blues', norm=np.log10)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for Multi-Class Classification - DNN')
    plt.show()