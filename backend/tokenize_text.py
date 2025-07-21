# run data_trim.py first

import pandas as pd
import re
import emoji
from contractions import fix
from nltk.corpus import stopwords

import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Normalize special characters
    text = re.sub(r'([!@#$%^&*()_+=\-[\]{};:\'",./<>?|\\~`])\1*', r'\1', text)
    # Convert emojis to text
    text = emoji.demojize(text)
    # Expand contractions
    text = fix(text)
    # Normalize slang
    text = re.sub(r'\bu\b', 'you', text)
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def tokenize_text(text):
    # Use regex to split into words while preserving special characters within words
    tokens = re.findall(r'\b\w+\b', text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

df = pd.read_csv('csv_data/trimmed_train.csv')

# Apply preprocessing to the 'comment_text' column
df['preprocessed_text'] = df['comment_text'].apply(preprocess_text)

# Apply tokenization to the preprocessed text
df['tokens'] = df['preprocessed_text'].apply(tokenize_text)

df.to_csv('csv_data/tokenized_train.csv', index=False)