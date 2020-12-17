#!/usr/bin/env python3

""" Create an inverted index from a document collection."""

from collections import defaultdict
from os.path import join

import pandas as pd
import spacy

from data_utils import get_article_content, save_index
from config import DATAPATH_PROCESSED


def create_inverted_index(dataframe, model):
    """ Create an invered index from a document collection."""
    inverted_index = defaultdict(set)
    stop_words = model.Defaults.stop_words
    # TODO: find a faster way to iterate; iterrows() is very slow
    total = len(dataframe)
    index2wikiid = {}
    for index, row in dataframe.iterrows():
        content = get_article_content(row["Text"], model, stop_words)
        wiki_key = row["Wikipedia_ID"]
        index2wikiid[index] = wiki_key
        for word in content:
            inverted_index[word].add(index)
        if index % 10000 == 0:
            print(index/total)
    final_index = {}
    for index, doc_ids in inverted_index.items():
        final_index[index] = list(doc_ids)
    return final_index, index2wikiid


if __name__ == "__main__":
    spacy_model = spacy.load('en_core_web_sm')
    df = pd.read_csv(join(DATAPATH_PROCESSED, "nq_dev_train_wiki_text.csv"))
    inverted_index, index2wikiid = create_inverted_index(df, spacy_model)
    save_index(inverted_index, filename="inverted_index.json")
    save_index(index2wikiid, filename="index2id.json")
