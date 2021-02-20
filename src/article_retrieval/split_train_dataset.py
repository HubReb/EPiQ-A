#!/usr/bin/env python3

"""Split quesion training set into training and dev set"""

from os.path import join

import pandas as pd

from article_retrieval.config import DATAPATH, DATAPATH_PROCESSED


def split(datapath: str, result_path: str):
    """Split question training set into training and dev set at 90/10"""
    df = pd.read_csv(join(datapath, "natural_questions_train.csv"))
    train, dev = df[:int(len(df)*0.9)], df[int(len(df)*0.9):]
    train.to_csv(join(result_path, "nq_train.csv"))
    dev.to_csv(join(result_path, "nq_dev.csv"))


if __name__ == "__main__":
    split(DATAPATH, DATAPATH_PROCESSED)
