#!/usr/bin/env python3

"""Fuse train and dev articles and calculate inverted index on resulting dataframe"""

from os.path import join

import pandas as pd

from article_retrieval.config import DATAPATH, DATAPATH_PROCESSED


def split(datapath: str, result_path: str):
    """ Fuse original training and dev set in one csv file """
    df1 = pd.read_csv(join(datapath, "nq_dev_wiki_text.csv"))
    df2 = pd.read_csv(join(datapath, "nq_train_wiki_text.csv"))
    df1 = pd.concat([df1, df2])
    df1.to_csv(join(result_path, "nq_dev_train_wiki_text.csv"))


if __name__ == "__main__":
    split(DATAPATH, DATAPATH_PROCESSED)
