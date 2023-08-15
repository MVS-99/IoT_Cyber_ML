def run_eda():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    
    # Create a dictionary to hold the abbreviations
    abbreviations = {}

    # Loading the preprocessed data
    df_train = pd.read_csv('CSV/preprocessed_DNN.csv', low_memory=False)

    # Displaying the first few rows of the DataFrame
    print(df_train.head())

    # Summary statistics of the DataFrame
    print(df_train.describe())

    # Checking for missing values
    print(df_train.isnull().sum())

    # Visualizing the distribution of the target variable ('Attack_type')
    # Get the frequency count of each attack type
    attack_counts_train = df_train['Attack_type'].value_counts()

    plt.figure(figsize=(15, 8))
    sns.set(style="whitegrid")
    sns.barplot(x=attack_counts_train.index, y=attack_counts_train.values, alpha=0.8, palette='viridis')
    plt.yscale("log")
    plt.title('Number of Attacks by Type (Log Scale)', fontsize=16)
    plt.ylabel('Number of Attacks (Log Scale)', fontsize=14)
    plt.xlabel('Attack Type', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()

    # Encoding 'Attack_type' to numerical categories
    label_encoder = LabelEncoder()
    df_train['Attack_type'] = label_encoder.fit_transform(df_train['Attack_type'])

    # Checking the correlation between features
    correlation = df_train.corr(method='spearman')

    # Selecting and visualizing the top 3 features with the highest correlation to the target variable
    correlation_target = abs(correlation['Attack_type'])
    top_correlations = correlation_target.nlargest(4)
    print('Top 3 features with the highest correlation to Attack_type:')
    print(top_correlations)

    # Visualizing the relationship between these top 3 features and the target variable using boxplots
    for feature in top_correlations.index[1:]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Attack_type', y=feature, data=df_train)
        plt.title(f'Relationship between Attack_type and {feature}')
        plt.xticks(rotation=90)
        plt.show()

    # Selecting and visualizing the top 3 features with the highest correlation to the target variable
    correlation_target_1 = abs(correlation['Attack_label'])
    top_correlations_1 = correlation_target_1.nlargest(4)
    print('Top 3 features with the highest correlation to Attack_label:')
    print(top_correlations_1)

    # Get the feature names of the top correlations, excluding 'Attack_label'
    top_correlations_1 = [feature for feature in top_correlations_1.index if feature != 'Attack_label']

    # Visualizing the relationship between these top 3 features and the target variable using boxplots
    for feature in top_correlations_1:
        # Create a combined feature
        df_train['combined'] = df_train[feature].astype(str) + "-" + df_train['Attack_label'].astype(str)

        plt.figure(figsize=(10, 6))
        sns.countplot(x='combined', data=df_train, order=['0-0', '0-1', '1-0', '1-1'])
        plt.title(f'Relationship between Attack_label and {feature}')
        plt.xlabel(f'{feature} - Attack_label')
        plt.xticks(rotation=90)
        plt.show()

    # Visualizing the correlation using a heatmap
    plt.figure(figsize=(20, 15))
    sns.set(style="whitegrid")

    heatmap = sns.heatmap(correlation, annot=False, cmap='coolwarm', linewidths=0.5)

    plt.title('Correlation Heatmap')
    # Rotate the x-axis and y-axis labels for better readability
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12)  # Added rotation for y-axis

    # Adjust the layout and remove unnecessary spines
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.show()