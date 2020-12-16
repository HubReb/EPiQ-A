#!/usr/bin/env python3

""" Create an inverted index from a document collection."""

from collections import defaultdict

import spacy
import pandas as pd

from data_utils import save_index, get_article_content


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
        if index % 100 == 0:
            print(index/total)
    final_index = {}
    for index, doc_ids in inverted_index.items():
        final_index[index] = list(doc_ids)
    return final_index, index2wikiid
