#!/usr/bin/env python3

"""Fuse train and dev articles and calculate inverted index on resulting dataframe"""

import spacy
import pandas as pd

from inverted_index import create_inverted_index
from data_utils import save_index


if __name__ == "__main__":
    model = spacy.load('en_core_web_sm')
    stop_words = model.Defaults.stop_words
    df1 = pd.read_csv("../../data/nq_dev_wiki_text.csv")
    df2 = pd.read_csv("../../data/nq_train_wiki_text.csv")
    df1 = pd.concat([df1, df2])
    df1.to_csv("../../data/nq_dev_train_wiki_text.csv")
    df = pd.read_csv("../../data/nq_dev_train_wiki_text.csv")
    inverted_index, index2wikiid = create_inverted_index(df, model)
    save_index(inverted_index, filename="dev_inverted_index.json")
    save_index(index2wikiid, filename="dev_index2id.json")
