# Sentiment Analysis on Tweets

This project performs sentiment analysis on tweets using the Sentiment140 dataset. The goal is to classify tweets as positive or negative based on their content.

## Project Structure

- `sentiment_analysis.py`: Main script to load, preprocess, and analyze the dataset.
- `training.csv`: The dataset file containing tweets and their sentiment labels.

## Requirements

- Python 3.6+
- Required libraries: `pandas`, `numpy`, `nltk`, `scikit-learn`

## Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/aaryanmittal/sentiment-analysis-tweets.git
    cd sentiment-analysis-tweets
    ```

2. **Install the required libraries**:
    ```bash
    pip install pandas numpy nltk scikit-learn
    ```

3. **Download NLTK stopwords**:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

3. **Download NLTK stopwords**:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

3. **Download the training database**:
    ```https://drive.google.com/file/d/1WgYbt_tiN6R4ZYx3RDO8BQZF-4vlcjmk/view?usp=drive_link ```

## Usage

1. **Ensure the dataset file is in the correct path**:
    - Place the `training.1600000.processed.noemoticon.csv` file in the same directory as `sentiment_analysis.py`.

2. **Run the sentiment analysis script**:
    ```bash
    python sentiment_analysis.py
    ```

3. **Output**:
    - The script will print the classification report and accuracy of the model.

## Explanation of the Script

- **Import Libraries**: The script imports necessary libraries for data manipulation, text processing, and machine learning.
- **Load Dataset**: The dataset is loaded using `pandas` with manually assigned column names.
- **Clean Text Data**: The text data is cleaned by removing URLs, mentions, special characters, and stopwords.
- **Feature Extraction**: The cleaned text is converted into numerical features using TF-IDF vectorization.
- **Split Data**: The dataset is split into training and testing sets.
- **Train Model**: A logistic regression model is trained on the training set.
- **Evaluate Model**: The model's performance is evaluated on the test set, and results are printed.

## Dataset

The dataset used in this project is Sentiment140, which contains 1,600,000 tweets with their corresponding sentiment labels. The sentiment labels are:
- `0` for negative sentiment
- `4` for positive sentiment


## Acknowledgments

- The Sentiment140 dataset was created by Alec Go, Richa Bhayani, and Lei Huang.
- Special thanks to the creators of the libraries and tools used in this project.
