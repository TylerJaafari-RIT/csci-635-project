import pandas as pd
import numpy as np
import sklearn.feature_extraction.text as skl_text
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import os

import sys

if len(sys.argv) > 1:
    fname = sys.argv[1]
else:
    fname = None

def import_data(fname = None):

    # Loading data
    if fname == None:
        fname = "../data/twitter_training.csv"
    train_set = pd.read_csv(fname, names=["Tweet Id","Entity","Sentiment","Tweet Content"])

    # Cleaning data
    ## Remove rows with missing values
    hasnan=train_set.isna().any(axis=1)
    for i in hasnan.index:
        if hasnan[i]:
            print("dropped",i)
            train_set=train_set.drop(i)

    print(train_set)
    train_set = train_set.reset_index(drop=True)
    print(train_set)


    # Fitting tf-idf vectorizer
    tfidf_vec = skl_text.TfidfVectorizer(max_features=200)
    tfidf_matrix = tfidf_vec.fit_transform(train_set["Tweet Content"])

    # Dimensionality reduction
    arr = [np.sum(tfidf_matrix[:,i]) for i in tqdm(range(tfidf_matrix.shape[1]))]

    plt.scatter(range(tfidf_matrix.shape[1]),arr)
    plt.show()

    names = tfidf_vec.get_feature_names_out()
    '''
    candidates = [[],[]]
    cutoff = 500
    for i in tqdm(range(len(arr))):
        if arr[i] > cutoff:
            candidates[0].append(names[i])
            candidates[1].append(arr[i])

    figure = plt.figure(figsize=[5,0.2*len(candidates[1])])
    plt.barh(candidates[0],candidates[1])
    figure.show()
    '''
    
    return train_set, tfidf_matrix
    
    
if __name__ == '__main__':
    if not os.path.exists("../data/clean/"):
        os.mkdir("../data/clean/")
    # clean and generate training and validation sets
    train_set, tfidf_matrix = import_data("../data/twitter_training.csv")
    train_set, val_set = train_test_split(train_set,test_size=1500)
    train_set.to_csv("../data/clean/twitter_training_clean.csv")
    val_set.to_csv("../data/clean/twitter_validation_clean.csv")

    # clean and generate test set
    test_set, tfidf_matrix = import_data("../data/twitter_validation.csv")
    test_set.to_csv("../data/clean/twitter_test_clean.csv")

    print(type(tfidf_matrix),tfidf_matrix)
