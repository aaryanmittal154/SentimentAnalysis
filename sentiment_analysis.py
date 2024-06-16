import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def load_data(filepath):
    column_names = ["sentiment", "id", "date", "query", "user", "text"]
    data = pd.read_csv(filepath, encoding="latin-1", header=None, names=column_names)
    print("Columns in the dataset:")
    print(data.columns)  # Print column names
    print("First few rows of the dataset:")
    print(data.head())  # Print the first few rows of the dataset

    return data


def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


def preprocess_data(data):
    data["cleaned_text"] = data["text"].apply(clean_text)
    return data


def extract_features(data):
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(data["cleaned_text"]).toarray()
    y = data["sentiment"]
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


def main():
    filepath = "/Users/aaryanmittal/ml1/training.csv"
    data = load_data(filepath)
    data = preprocess_data(data)
    X, y = extract_features(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
