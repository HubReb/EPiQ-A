#!/usr/bin/env python3


""" Evaluate complete pipeline end to end """

import time

from pipeline import CombinedModel
from article_retrieval.config import BM25_MODEL, TFIDF_MODEL

from helper_functions import (
        load_csv,
        normalize,
        jaccard_index,
        bleu,
        word_error_rate,
        exact_match,
        f1
)


def evaluate(model):
    """Evaluation main"""
    # model.get_answer("who had most wins in nfl")
    # print()
    # model = CombinedModel("bm", "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv", BM25_MODEL)
    # model.get_answer("who had most wins in nfl")

    print("Loading questions")
    question_dev_dataframe = load_csv("../data/natural_questions_dev.csv")

    print("Predicting answers...")

    sum_jaccard_index = 0.0
    sum_bleu = 0.0
    sum_wer = 0.0
    sum_exact_match = 0.0
    sum_f1 = 0.0
    sum_time = 0.0

    print("Starting evaluation")
    print()
    for i, row in question_dev_dataframe.iterrows():
        starttime = time.time()
        question = row["Question"]
        predicted_answer = model.get_answer(question)
        endtime = time.time()

        correct_answer = row["Answer"]

        sum_jaccard_index += jaccard_index(
            normalize(predicted_answer).split(), normalize(correct_answer).split()
        )
        sum_bleu += bleu(predicted_answer, correct_answer)
        sum_wer += word_error_rate(
            normalize(predicted_answer).split(), normalize(correct_answer).split()
        )
        sum_exact_match += exact_match(
            normalize(predicted_answer).split(), normalize(correct_answer).split()
        )
        sum_f1 += f1(
            normalize(predicted_answer).split(), normalize(correct_answer).split()
        )
        sum_time += endtime - starttime

        print(" " * 180, end="\r")
        print(
            "Avg. Jaccard-Index: {:.2f}\tAvg. BLEU: {:.2f}\tAvg. WER: {:.2f}\tAvg. Exact Match: {:.2f}\tAvg. F1: {:.2f}\tAvg. Time/Question: {:.2f}s\tTotal questions: {}".format(
                sum_jaccard_index / (i + 1),
                sum_bleu / (i + 1),
                sum_wer / (i + 1),
                sum_exact_match / (i + 1),
                sum_f1 / (i + 1),
                sum_time / (i + 1),
                i + 1,
            ),
            end="\r",
        )


if __name__ == "__main__":
    print("Loading model")
    test_model = CombinedModel(
        "tfidf", TFIDF_MODEL, article_retrieval_use_index=True, retrieve_articles=True
    )
    evaluate(test_model)
