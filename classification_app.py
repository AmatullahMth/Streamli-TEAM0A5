import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
import Streamlit as st

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tag import pos_tag
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from spellchecker import Spellcheck


spell = Spellcheck()
stop_words = stopwords.words('english')

# Ensure the necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')


vectorizer = load_pickle("tfidf_vectorizer.pkl")
le = load_pickle("LabelEncoder.pkl")
modelNB = load_pickle("MultinomialNB.pkl")

# Function to clean the text data
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def df_processor(dataframe):
    df = dataframe.copy()
    df['all_text'] = df['all_text'].apply(clean_text)
    return df

def main():
    st.title("Text Data Classification App")

    # Input text box
    input_text = st.text_area("Enter News Text:", height=200)

    # Model selection
    model_choice = st.selectbox(
        "Select a classification model:",
        ("MultinomialNB", "Random Forest", "Logistic Regression")
    )

    st.write("You selected:", model_choice)

    # Create the classifier instances
    modelRF = RandomForestClassifier()
    modelLR = LogisticRegression()

    # Classify button
    if st.button("Classify"):
        if input_text.strip() == "":
            st.warning("Please enter news article.")
        else:
            # Convert user text into dataframe
            userdf = pd.DataFrame({'all_text': [input_text]})

            # Perform preprocessing
            cleandf = df_processor(userdf)

            # Convert to features using the vectorizer
            X_ft = vectorizer.transform(cleandf['all_text'])

            # Perform a prediction on the vectorized text
            if model_choice == "MultinomialNB":
                y_pred = modelNB.predict(X_ft)
            elif model_choice == "Random Forest":
                y_pred = modelRF.predict(X_ft)
            elif model_choice == "Logistic Regression":
                y_pred = modelLR.predict(X_ft)

            # Inverse transform and print the category
            y_tran = le.inverse_transform(y_pred)
            readable_cat = y_tran[0].title()

            # Output to the user
            st.success(readable_cat)

if __name__ == "__main__":
    main()
