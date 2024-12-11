# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (only needed once)
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset (ensure your CSV file has a 'text' column and 'label' column)
# Assuming the dataset is in a CSV file 'spam.csv'
# The 'label' column should contain 'ham' for non-spam and 'spam' for spam emails
df = pd.read_csv("spam.csv")

# Data Preprocessing
# Remove any rows with missing data in 'text' or 'label' columns
df.dropna(subset=["text", "label"], inplace=True)

# Tokenize the text and remove stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    This function takes in an email text, tokenizes it, converts it to lowercase,
    removes stopwords and non-alphabetic tokens, and returns the cleaned text.
    """
    tokens = word_tokenize(text.lower())  # Lowercase and tokenize
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]  # Remove stopwords and non-alphabetic tokens
    return " ".join(filtered_tokens)

# Apply preprocessing to the 'text' column
df['text'] = df['text'].apply(preprocess_text)

# Convert 'label' to binary: 0 for 'ham' (non-spam) and 1 for 'spam'
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split dataset into features (X) and target (y)
X = df['text']  # Feature: email text
y = df['label']  # Target: spam or ham

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF (Term Frequency - Inverse Document Frequency)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes model (Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example of making a sample prediction
sample_text = ["Congratulations! You've won a $1000 gift card, click here to claim."]
sample_tfidf = vectorizer.transform(sample_text)  # Transform sample text into the same format as the training data
sample_prediction = model.predict(sample_tfidf)

if sample_prediction[0] == 1:
    print("\nThe email is classified as SPAM.")
else:
    print("\nThe email is classified as HAM (non-spam).")