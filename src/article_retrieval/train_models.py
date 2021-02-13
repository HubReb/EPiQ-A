#!/usr/bin/env python3

"""Train ranking model with Okapi BM 25 weights and ranking model with Tf-IDF weights."""

import pickle
from os.path import join


from article_retrieval.gensim_bm25 import Okapi25
from article_retrieval.tf_idfs import TFIDFmodel
from article_retrieval.config import DATAPATH_PROCESSED
from article_retrieval.article_index import ArticlesFromTitleMentions


def train(datapath):
    """Train okapi BM25 and TFIDF models"""
    bm = Okapi25(1.5, 0.75, 0.25)
    bm.fit(join(datapath, "nq_dev_train_wiki_text_merged.csv"))
    with open("okapibm25.pkl", "wb") as f:
        pickle.dump(bm, f)
    tfidf_model = TFIDFmodel()
    tfidf_model.create_tf_idf_vectors(join(datapath, "nq_dev_train_wiki_text_merged.csv"))
    with open("tfidfmodel.pkl", "wb") as f:
        pickle.dump(tfidf_model, f)
    
    article_getter = ArticlesFromTitleMentions(join(datapath, 'nq_dev_train_wiki_text.csv'))
    article_getter.save()


if __name__ == "__main__":
    train(DATAPATH_PROCESSED)
