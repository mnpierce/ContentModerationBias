# Run tokenize_text.py first

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('csv_data/tokenized_train.csv')

# Ensure the 'tokens' column is a list of strings
df['tokens'] = df['tokens'].apply(eval)  

# Join tokens into a single string
df['text'] = df['tokens'].apply(lambda x: ' '.join(x))

tfidf_vectorizer = TfidfVectorizer(max_features=10000)  # You can adjust max_features
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

# Split the data into training and testing sets
X = tfidf_matrix  # Features (TF-IDF matrix)
df['binary_target'] = (df['target'] >= 0.5).astype(int)
y = df['binary_target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
print("Logistic Regression\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Train a Multinomial NB model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
print("Multinomial Bayes\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Train a Complement NB model (theoretically better at handling imbalance)
model = ComplementNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
print("ComplementNB\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))