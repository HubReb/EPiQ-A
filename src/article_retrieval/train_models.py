#!/usr/bin/env python3

"""Train ranking model with Okapi BM 25 weights and ranking model with Tf-IDF weights."""

import pickle

import spacy

from gensim_bm25 import Okapi25
from query_index import query_index
from tf_idfs import TFIDFmodel
from data_utils import read_index


if __name__ == "__main__":
    model = spacy.load('en_core_web_sm')
    bm = Okapi25(1.5, 0.75, 0.25)
    bm.fit("../../data/nq_dev_train_wiki_text.csv")
    with open("okapibm25.pkl", "wb") as f:
        pickle.dump(bm, f)
    index = read_index("inverted_index.json")
    stop_words = model.Defaults.stop_words
    docs, query = query_index("Who was George Bush?", index, model)
    tfidf_model = TFIDFmodel()
    tfidf_model.create_tf_idf_vectors("../../data/nq_dev_train_wiki_text.csv")
    with open("tfidfmodel.pkl", "wb") as f:
        pickle.dump(tfidf_model, f)
