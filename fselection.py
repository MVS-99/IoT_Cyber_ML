# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def select_features(df, target_column, exclude_column, num_features):
    """
    Select features based on their importance for predicting whether an attack occurs.
    """
    # Ensure target column is numeric
    if df[target_column].dtype == 'object':
        df[target_column] = df[target_column].astype('category').cat.codes

    # Define features
    features = df.drop([target_column, exclude_column], axis=1)

    # Fit a random forest classifier to the data
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, df[target_column])

    # Get feature importances
    importances = clf.feature_importances_

    # Create a DataFrame of features and importances
    features_importances = pd.DataFrame({'Feature': features.columns, 'Importance': importances})

    # Sort the DataFrame by importance in descending order and select top features
    features_importances = features_importances.sort_values(by='Importance', ascending=False).head(num_features)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=features_importances)
    plt.xscale('log')
    if target_column == "Attack_label":
        plt.title('Feature Importances for Attack Detection')
    else:
        plt.title('Feature Importances for Attack Type Classification')
    plt.show()

    # Create a new dataframe containing only the most important features
    df = df[features_importances['Feature'].tolist() + [target_column]]

    return df