#!/usr/bin/env python3

"""Split quesion training set into training and dev set"""

import pandas as pd


def main():
    """Split question training set into training and dev set at 90/10"""
    df = pd.read_csv("../../data/natural_questions_train.csv")
    train, dev = df[:int(len(df)*0.9)], df[int(len(df)*0.9):]
    train.to_csv("../../data/nq_train.csv")
    dev.to_csv("../../data/nq_dev.csv")


if __name__ == "__main__":
    main()
