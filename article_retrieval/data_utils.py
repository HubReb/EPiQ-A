#!/usr/bin/env python3

import json

import pandas as pd


def get_data(filename):
    df = pd.read_csv(filename)
    return df


def read_index(filename):
    with open(filename) as f:
        return json.load(f)


def save_index(index, filename="inverted_index.json"):
    with open(filename, "w") as f:
        json.dump(index, f)


if __name__ == "__main__":
    df = get_data("../data/natural_questions_train.csv")
    print(df.head())
    df2 = get_data("../data/nq_train_wiki_text.csv")
    print(df2.head())
