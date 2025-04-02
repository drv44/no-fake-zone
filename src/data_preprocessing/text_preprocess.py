# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from typing import List

# # Make sure to download NLTK resources if you haven't already:
# nltk.download('stopwords')
# nltk.download('punkt')

# def load_text_data(csv_path: str) -> pd.DataFrame:
#     """
#     Loads the CSV file containing text data.
#     Args:
#         csv_path (str): Path to the CSV file (e.g., 'data/text/merged_data.csv').
#     Returns:
#         pd.DataFrame: DataFrame with columns like ['text', 'subject', 'date', 'label'].
#     """
#     df = pd.read_csv(csv_path)
#     return df

# def clean_text(text: str) -> str:
#     """
#     Cleans a single text string by removing unwanted characters, links, etc.
#     Args:
#         text (str): Raw text.
#     Returns:
#         str: Cleaned text.
#     """
#     # Remove URLs
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
#     # Remove non-alphanumeric characters (keeping basic punctuation for meaning)
#     text = re.sub(r"[^a-zA-Z0-9\s\.,!?']", '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     return text

# def tokenize_and_remove_stopwords(text: str) -> List[str]:
#     """
#     Tokenizes text and removes stopwords.
#     Args:
#         text (str): Cleaned text string.
#     Returns:
#         List[str]: List of tokens (words).
#     """
#     stop_words = set(stopwords.words('english'))
#     tokens = word_tokenize(text)
#     filtered_tokens = [word for word in tokens if word not in stop_words]
#     return filtered_tokens

# def preprocess_text_column(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
#     """
#     Applies cleaning and tokenization to a specified text column in a DataFrame.
#     Args:
#         df (pd.DataFrame): DataFrame containing the text data.
#         text_column (str): The name of the column containing text.
#     Returns:
#         pd.DataFrame: Updated DataFrame with a new column 'processed_text'.
#     """
#     df['cleaned_text'] = df[text_column].apply(clean_text)
#     df['tokens'] = df['cleaned_text'].apply(tokenize_and_remove_stopwords)
#     return df

# def main():
#     # Example usage:
#     csv_path = 'data/text_data/shuffled_merged.csv'
#     df = load_text_data(csv_path)
    
#     # Basic cleaning + tokenization
#     df = preprocess_text_column(df, text_column='text')
    
#     # Now you have:
#     #   - df['cleaned_text'] containing the cleaned text
#     #   - df['tokens'] containing the list of tokens
#     print(df.head())

# if __name__ == "__main__":
#     main()

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')

def load_text_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the CSV file containing text data.
    """
    df = pd.read_csv(csv_path)
    return df

def clean_text(text) -> str:
    """
    Cleans a single text string by removing unwanted characters, links, etc.
    Converts non-string values to an empty string.
    """
    if not isinstance(text, str):
        text = ""
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # Remove non-alphanumeric characters (keeping basic punctuation for meaning)
    text = re.sub(r"[^a-zA-Z0-9\s\.,!?']", '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def tokenize_and_remove_stopwords(text: str) -> List[str]:
    """
    Tokenizes text and removes stopwords.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def preprocess_text_column(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Applies cleaning and tokenization to a specified text column in a DataFrame.
    """
    df['cleaned_text'] = df[text_column].apply(clean_text)
    df['tokens'] = df['cleaned_text'].apply(tokenize_and_remove_stopwords)
    return df

def main():
    csv_path = 'data/text_data/shuffled_merged.csv'
    df = load_text_data(csv_path)
    
    # Apply text preprocessing
    df = preprocess_text_column(df, text_column='text')
    
    # Display a sample of processed data
    print(df.head())

if __name__ == "__main__":
    main()
