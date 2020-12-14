#!/usr/bin/env python3


import json
import pickle

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


from data_utils import get_data
from inverted_index import get_article_content


def create_tf_idf_vectors(dataframe):
    index2key = {}
    index2vector = {}
    content_matrix = []
    model = spacy.load("en_core_web_sm")
    stop_words = model.Defaults.stop_words
    for index, row in dataframe.iterrows():
        content = " ".join(get_article_content(row["Text"], model, stop_words))
        index2key[index] = row["Wikipedia_ID"]
        content_matrix.append(content)
    # stop words are already removed
    vectorizer = TfidfVectorizer(min_df=5, strip_accents="unicode")
    doc_vecs = vectorizer.fit_transform(content_matrix)
    for index, vector in enumerate(doc_vecs):
        index2vector[index2key[index]] = vector
    with open("tf_idf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    return index2vector


df = get_data("../data/nq_train_wiki_text.csv")
tf_idf_vectors = create_tf_idf_vectors(df)
with open("document_vectors.json", "w") as f:
    json.dump(tf_idf_vectors, f)
