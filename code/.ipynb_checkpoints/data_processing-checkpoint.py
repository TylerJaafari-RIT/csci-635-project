import pandas as pd
import numpy as np
import pickle as pickle
import sklearn.feature_extraction.text as skl_text
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

def import_data():

    # Loading data
    train_set = pd.read_csv("../data/twitter_training.csv",names=["Tweet Id","Entity","Sentiment","Tweet Content"])

    # Cleaning data
    ## Remove rows with missing values
    hasnan=train_set.isna().any(axis=1)
    for i in hasnan.index:
        if hasnan[i]:
            print("dropped",i)
            train_set=train_set.drop(i)

    # Fitting tf-idf vectorizer
    tfidf_vec = skl_text.TfidfVectorizer(max_features=200)
    tfidf_matrix = tfidf_vec.fit_transform(train_set["Tweet Content"])

    # Dimensionality reduction
    arr = [np.sum(tfidf_matrix[:,i]) for i in tqdm(range(tfidf_matrix.shape[1]))]

    plt.scatter(range(tfidf_matrix.shape[1]),arr)

    names = tfidf_vec.get_feature_names_out()
    candidates = [[],[]]
    cutoff = 500
    for i in tqdm(range(len(arr))):
        if arr[i] > cutoff:
            candidates[0].append(names[i])
            candidates[1].append(arr[i])

    figure = plt.figure(figsize=[5,0.2*len(candidates[1])])
    plt.barh(candidates[0],candidates[1])
    figure.show()

    return train_set, tfidf_vec, tfidf_matrix