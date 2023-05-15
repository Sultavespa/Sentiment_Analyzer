import requests
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Your Twitter API keys
bearer_token = 'your bearer token'

# Load the trained model and vectorizer
nb_model = joblib.load('nb_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Define a function to clean the tweets
def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
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

# Define a function to fetch tweets
def fetch_tweets(query, max_results=100):
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    query_params = {
        'query': query,
        'tweet.fields': 'created_at,lang',
        'max_results': max_results
    }
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": "v2RecentSearchPython"
    }
    response = requests.get(search_url, headers=headers, params=query_params)
    if response.status_code != 200:
        raise Exception(f"Request returned an error: {response.status_code}, {response.text}")
    return response.json()

# Fetch tweets
tweets_data = fetch_tweets('AI')

# Extract tweet texts
tweets = [tweet['text'] for tweet in tweets_data['data']]

# Clean the tweets
cleaned_tweets = clean_tweets(tweets)

# Convert tweets to vectors
tweet_vectors = vectorizer.transform(cleaned_tweets)

# Predict sentiment
predictions = nb_model.predict(tweet_vectors)

# Print predictions
for tweet, sentiment in zip(tweets, predictions):
    print(f"{tweet} --> {'Positive' if sentiment == 1 else 'Negative'}")
