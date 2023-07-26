# Import necessary libraries
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.neighbors import LocalOutlierFactor

def select_features(df, target_column, k):
    """
    Select the k best features from df based on their relationship with the target_column.
    """
    # Convert categorical target variable to numerical if it's not
    df[target_column] = df[target_column].astype('category').cat.codes

    # Select top k features
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(df.drop(target_column, axis=1), df[target_column])

    # Get the names of the selected features
    selected_features = df.drop(target_column, axis=1).columns[selector.get_support()]

    return df[selected_features]

def detect_and_remove_outliers(df, n_neighbors, contamination):
    """
    Use the Local Outlier Factor method to detect and remove outliers from df.
    """
    # Use Local Outlier Factor to identify outliers
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(df)

    # Filter out the outliers
    mask = y_pred != -1
    df = df[mask]

    return df