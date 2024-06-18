import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns
import os

# Set plot style
sns.set(style='whitegrid')

# File paths
file_paths = [
    'C:/Users/nisch/Downloads/employee_review_mturk_dataset_test_v6_kaggle.csv',
    'C:/Users/nisch/Downloads/employee_review_mturk_dataset_v10_kaggle.csv',
    'C:/Users/nisch/Downloads/test_set.csv',
    'C:/Users/nisch/Downloads/train_set.csv',
    'C:/Users/nisch/Downloads/validation_set.csv'
]

# List to store DataFrames
dfs = []

# Load CSV files
for file_path in file_paths:
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
            if not df.empty:
                dfs.append(df)
                print(f"Loaded {file_path} successfully.")
            else:
                print(f"{file_path} is empty.")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    else:
        print(f"{file_path} does not exist.")

# Combine DataFrames
if dfs:
    df_combined = pd.concat(dfs, ignore_index=True)
    print("Combined DataFrame info:")
    print(df_combined.info())

    # Drop duplicate entries
    df_combined.drop_duplicates(inplace=True)

    # Drop rows with missing values in critical columns
    df_combined.dropna(subset=['feedback'], inplace=True)

    # Fill missing values in less critical columns
    df_combined['label'].fillna(df_combined['label'].mean(), inplace=True)
    df_combined['performance_class'].fillna(df_combined['performance_class'].mean(), inplace=True)
    df_combined['potential_class'].fillna(df_combined['potential_class'].mean(), inplace=True)
    df_combined['feedback_clean'].fillna('', inplace=True)
    df_combined['feedback_len'].fillna(df_combined['feedback_len'].mean(), inplace=True)
    df_combined['num_of_sent'].fillna(df_combined['num_of_sent'].mean(), inplace=True)

    # Display descriptive statistics for numerical columns
    print(df_combined.describe())

    # Visualize Response Distribution
    # Assuming there's a 'rating' column for numeric ratings
    if 'rating' in df_combined.columns:
        plt.figure(figsize=(10, 6))
        df_combined['rating'].hist(bins=10)
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.title('Distribution of Ratings')
        plt.show()

    # Generate Word Cloud for Open-Ended Responses
    # Assuming there's a 'feedback' column for textual feedback
    if 'feedback' in df_combined.columns:
        text = ' '.join(df_combined['feedback'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Employee Feedback')
        plt.show()

    # Sentiment Analysis
    # Sentiment Analysis on 'feedback' column
    if 'feedback' in df_combined.columns:
        df_combined['sentiment'] = df_combined['feedback'].apply(lambda x: TextBlob(x).sentiment.polarity)
        plt.figure(figsize=(10, 6))
        sns.histplot(df_combined['sentiment'], bins=20, kde=True)
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sentiment Polarity')
        plt.show()

        # Display basic sentiment statistics
        print(df_combined['sentiment'].describe())

    # Thematic Analysis
    # Assuming the feedback has been categorized or themes have been extracted, visualize the themes
    # This step would generally involve more advanced text analysis techniques, potentially using NLP libraries like spaCy or NLTK

    # For example, if you have a 'theme' column:
    if 'theme' in df_combined.columns:
        theme_counts = df_combined['theme'].value_counts()

        plt.figure(figsize=(12, 8))
        sns.barplot(x=theme_counts.index, y=theme_counts.values)
        plt.xlabel('Theme')
        plt.ylabel('Count')
        plt.title('Distribution of Themes in Feedback')
        plt.xticks(rotation=45)
        plt.show()
else:
    print("No data loaded from the provided file paths.")