#!/usr/bin/env python3

"""Fuse train and dev articles and calculate inverted index on resulting dataframe"""

from os.path import join

import spacy
import pandas as pd

from inverted_index import create_inverted_index
from data_utils import save_index
from config import DATAPATH, DATAPATH_PROCESSED


def split(datapath, result_path):
    model = spacy.load('en_core_web_sm')
    df1 = pd.read_csv(join(datapath, "nq_dev_wiki_text.csv"))
    df2 = pd.read_csv(join(datapath, "nq_train_wiki_text.csv"))
    df1 = pd.concat([df1, df2])
    df1.to_csv(join(result_path, "nq_dev_train_wiki_text.csv"))
    df = pd.read_csv(join(result_path, "nq_dev_train_wiki_text.csv"))
    inverted_index, index2wikiid = create_inverted_index(df, model)
    save_index(inverted_index, filename="inverted_index.json")
    save_index(index2wikiid, filename="index2id.json")


if __name__ == "__main__":
    split(DATAPATH, DATAPATH_PROCESSED)
