# Script 1: train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

nltk.download('punkt')
nltk.download('stopwords')

# Load Sentiment140 dataset
print("Loading Sentiment140 dataset...")
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])

# Map target labels to 0 (negative) and 1 (positive)
df['target'] = df['target'].map({0: 0, 4: 1})

# Define a function to clean the tweets
def clean_tweets(tweets):
    counter = 0
    cleaned_tweets = []
    for tweet in tweets:
        counter=counter+1
        print(str(counter)+": cleaning tweet: "+tweet)
        # Lowercase
        tweet = tweet.lower()
        # Remove URLs
        tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
        # Remove punctuations
        tweet = tweet.translate(str.maketrans('', '', string.punctuation))
        # Tokenize the tweet
        tweet_tokens = word_tokenize(tweet)
        # Remove stopwords
        filtered_words = [word for word in tweet_tokens if word not in stopwords.words('english')]
        cleaned_tweets.append(" ".join(filtered_words))
    return cleaned_tweets

# Clean the tweets
print("Cleaning the tweets...")
cleaned_tweets = clean_tweets(df['text'])

# Split the data into training and test sets
print("Splitting the data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(cleaned_tweets, df['target'], test_size=0.2, random_state=42)

# Initialize a TF-IDF vectorizer and transform the data
print("Vectorizing the data...")
vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train a Naive Bayes model
print("Training a Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_vectors, y_train)

# Predict on the test set and print a classification report
print("Predicting on the test set...")
y_pred = nb_model.predict(X_test_vectors)
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
print("Saving the trained model and vectorizer...")
joblib.dump(nb_model, 'nb_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
print("DONE")