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

def process_file(fname):
    if not os.path.exists(fname):
        raise FileNotFoundError("Clean data files not found. Run data_processing.py and balance_data.py first.")
    data_set = pd.read_csv(fname)
    # Fitting tf-idf vectorizer
    tfidf_vec = skl_text.TfidfVectorizer(min_df=0.002, max_df=0.6)
    tfidf_matrix = tfidf_vec.fit_transform(data_set["Tweet Content"])

    entenc = skl_pre.LabelBinarizer()
    entity = entenc.fit_transform(data_set['Entity'].to_xarray())
    sentenc = skl_pre.LabelEncoder()
    sentiment = sentenc.fit_transform(data_set['Sentiment'])

    class_names = sentenc.inverse_transform(range(4))
    pd.DataFrame(class_names).to_csv("../data/class_names.csv")

    X, y = np.concatenate((entity,tfidf_matrix.toarray()), axis=1), sentiment

    return X, y

def get_vectorized_data():
    X_train, y_train = process_file(target_training_file)
    X_val, y_val = process_file(target_validation_file)
    X_test, y_test = process_file(target_test_file)

    return X_train, y_train, X_val, y_val, X_test, y_test