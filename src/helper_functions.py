#!/usr/bin/env python3

""" Useful functions for evaluation and pipeline construction """

from collections import Counter

import sacrebleu
import editdistance
import pandas as pd


def load_csv(path: str):
    return pd.read_csv(path)


def normalize(text):
    return text.lower()


def jaccard_index(predicted, correct):
    predicted, correct = set(predicted), set(correct)
    union = len(set.union(predicted, correct))
    intersection = len(set.intersection(predicted, correct))

    return intersection / union


def bleu(predicted, correct):
    return sacrebleu.sentence_bleu(predicted, [correct]).score


def word_error_rate(predicted, correct):
    error_rate = editdistance.eval(predicted, correct)
    error_rate /= len(correct)
    return error_rate


def exact_match(predicted, correct):
    return float(int(predicted == correct))


def f1(predicted, correct):
    common_tokens = Counter(predicted) & Counter(correct)
    count_common = sum(common_tokens.values())
    if not count_common:
        f1 = 0.0
    else:
        precision = 1.0 * count_common / len(predicted)
        recall = 1.0 * count_common / len(correct)
        f1 = (2 * precision * recall) / (precision + recall)

    return f1


