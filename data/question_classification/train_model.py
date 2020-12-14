import spacy
import nltk
import random
import numpy as np

import gensim.downloader as gensim

from collections import namedtuple
from functools import partial
from typing import List, Union
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


TRAINFILE = "train.txt"
TESTFILE = "test.txt"
Stopwords = set(stopwords.words('english'))
EmbeddingModel = gensim.load("glove-twitter-25")

def load_data_from_file(filename):
    # Load train
    X_text = []
    y = []
    with open(filename) as tf:
        for line in tf:
            parts = line.strip().split()
            label = parts[0].strip()
            tokens = [token.lower() for token in parts[1:]]
            tokens = [token for token in tokens if token not in Stopwords]
            X_text.append(tokens)
            y.append(label)
    
    X = []
    for question in X_text:
        embeddings = []
        for token in question:
            try:
                embeddings.append(EmbeddingModel[token])
            except KeyError:
                continue
        
        if len(embeddings) > 0:
            X.append(np.max(np.stack(embeddings), axis=0))
        else:
            X.append(np.zeros(EmbeddingModel.vector_size))
    
    return np.stack(X), y


def load_data():
    X_train, y_train = load_data_from_file(TRAINFILE)
    X_test, y_test = load_data_from_file(TESTFILE)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    return classifier


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    
    print("Accuracy on training set: {:.2f}".format(accuracy_score(y_train, model.predict(X_train))))
    print("Accuracy on test set: {:.2f}".format(accuracy_score(y_test, model.predict(X_test))))
    
    
    
    
            
