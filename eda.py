def run_eda():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    
    # Loading the preprocessed data
    df = pd.read_csv('preprocessed_DNN.csv', low_memory=False)

    # Displaying the first few rows of the DataFrame
    print(df.head())

    # Summary statistics of the DataFrame
    print(df.describe())

    # Checking for missing values
    print(df.isnull().sum())

    # Visualizing the distribution of the target variable ('Attack_type')
    # Get the frequency count of each attack type
    attack_counts = df['Attack_type'].value_counts()

    plt.figure(figsize=(15, 8))
    sns.set(style="whitegrid")
    sns.barplot(x=attack_counts.index, y=attack_counts.values, alpha=0.8, palette='viridis')
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
    df['Attack_type'] = label_encoder.fit_transform(df['Attack_type'])

    # Checking the correlation between features
    correlation = df.corr()

    # Visualizing the correlation using a heatmap
    plt.figure(figsize=(11.7, 8.27))
    sns.set(style="whitegrid")
    heatmap = sns.heatmap(correlation, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    # Rotate the x-axis labels for better readability
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=12)
    # Adjust the layout and remove unnecessary spines
    plt.tight_layout()
    sns.despine(left=True, bottom=True)
    plt.show()

    # Selecting and visualizing the top 3 features with the highest correlation to the target variable
    correlation_target = abs(correlation['Attack_type'])
    top_correlations = correlation_target.nlargest(4)
    print('Top 3 features with the highest correlation to Attack_type:')
    print(top_correlations)

    # Visualizing the relationship between these top 3 features and the target variable using boxplots
    for feature in top_correlations.index[1:]:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Attack_type', y=feature, data=df)
        plt.title(f'Relationship between Attack_type and {feature}')
        plt.xticks(rotation=90)
        plt.show()

        # Selecting and visualizing the top 3 features with the highest correlation to the target variable
    correlation_target_1 = abs(correlation['Attack_label'])
    top_correlations_1 = correlation_target_1.nlargest(4)
    print('Top 3 features with the highest correlation to Attack_label:')
    print(top_correlations_1)

    # Visualizing the relationship between these top 3 features and the target variable using boxplots
    for feature in top_correlations_1:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue='Attack_label', data=df)
        plt.title(f'Relationship between Attack_label and {feature}')
        plt.xticks(rotation=90)
        plt.show()