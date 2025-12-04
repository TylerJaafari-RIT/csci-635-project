import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as skl_text
import sklearn.preprocessing as skl_pre
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import sys

target_training_file = "../data/clean/twitter_train_balanced.csv"
target_validation_file = "../data/clean/twitter_validation_clean.csv"
target_test_file = "../data/clean/twitter_test_clean.csv"

def vectorize(data_set, entenc, sentenc, tfidf_vec):
    entity = entenc.transform(data_set["Entity"].to_xarray())
    sentiment = sentenc.transform(data_set["Sentiment"])
    tfidf_matrix = tfidf_vec.transform(data_set["Tweet Content"])
    X, y = np.concatenate((entity, tfidf_matrix.toarray()), axis=1), sentiment

    return X, y

def get_vectorized_data():
    if not os.path.exists(target_training_file):
        raise FileNotFoundError("Clean data files not found. Run data_processing.py and balance_data.py first.")
    train_set = pd.read_csv(target_training_file)
    val_set = pd.read_csv(target_validation_file)
    test_set = pd.read_csv(target_test_file)
    # Fitting tf-idf vectorizer
    tfidf_vec = skl_text.TfidfVectorizer(min_df=0.002, max_df=0.6)
    tfidf_vec.fit(train_set["Tweet Content"])

    entenc = skl_pre.LabelBinarizer()
    sentenc = skl_pre.LabelEncoder()

    entenc.fit(train_set["Entity"].to_xarray())
    sentenc.fit(train_set["Sentiment"])

    class_names = sentenc.inverse_transform(range(4))
    pd.DataFrame(class_names).to_csv("../data/class_names.csv")

    X_train, y_train = vectorize(train_set, entenc, sentenc, tfidf_vec)
    X_val, y_val = vectorize(val_set, entenc, sentenc, tfidf_vec)
    X_test, y_test = vectorize(test_set, entenc, sentenc, tfidf_vec)

    return X_train, y_train, X_val, y_val, X_test, y_test