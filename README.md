# Sentiment Analyzer

This Sentiment Analyzer is a Python-based application that performs sentiment analysis on text data, specifically tweets. It utilizes Natural Language Processing (NLP) techniques and machine learning algorithms to classify tweets as positive, negative, or neutral.

## How It Works

1. Data Gathering: The application uses the Twitter API to fetch tweets based on a given query. The fetched tweets are stored in a CSV file. You need to have Twitter's V2 Api and put your bearer token in the code.

2. Data Preprocessing: The tweets are cleaned by removing URLs, punctuation, and stopwords. The remaining text is tokenized and transformed into numerical features.

3. Feature Extraction: The application uses the TF-IDF (Term Frequency-Inverse Document Frequency) technique to convert the text data into numerical feature vectors.

4. Model Selection: The model used for sentiment analysis is a Naive Bayes classifier, specifically the Multinomial Naive Bayes algorithm.

5. Training the Model: The model is trained using a labeled dataset, such as the Sentiment140 dataset, which contains tweets labeled as positive or negative. The dataset is preprocessed, and the features are extracted using TF-IDF. The trained model is then saved for later use.

6. Predicting Sentiment: The application allows you to input new tweets or text data and predicts the sentiment (positive, negative, or neutral) using the trained model.

## Libraries Used

The Sentiment Analyzer project utilizes the following libraries:

- Tweepy: For accessing the Twitter API and fetching tweets.
- NLTK (Natural Language Toolkit): For text preprocessing tasks such as tokenization, stopwords removal, and word normalization.
- Scikit-learn: For feature extraction using TF-IDF and training the Naive Bayes classifier.
- Pandas: For data manipulation and handling the fetched tweets.
- Joblib: For saving and loading the trained model and vectorizer.

Feel free to explore the code and customize it to suit your needs.

Happy sentiment analysis!
