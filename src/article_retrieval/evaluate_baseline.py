#!/usr/bin/env python3

""" Evaluation script for baselines """

import json
import pickle
from os.path import join

import spacy
import pandas as pd

from query_index import query_index, query_processing
from gensim_bm25 import Okapi25

from config import DATAPATH_PROCESSED, DATAPATH


def load_from_pickle(filename):
    """Load dataset from pickled file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_from_json(filename):
    """Load dictionary from json file."""
    with open(filename) as f:
        return json.load(f)


def load_utilities_for_bm(bm_file="okapibm25.pkl", index_file="inverted_index.json"):
    """Load Okapi25 model and inverted index from their respective files."""
    bm25 = load_from_pickle(bm_file)
    inverted_index = load_from_json(index_file)
    return bm25, inverted_index


def load_utilities_for_tfidf(
    model_file="tfidfmodel.pkl",
    index_file="inverted_index.json"
):
    """Load TFIDFmodel model and inverted index from their respective files."""
    tfidf_model = load_from_pickle(model_file)
    inverted_index = load_from_json(index_file)
    return tfidf_model, inverted_index


def load_dataset(filename, model):
    """Load dataset from csv file and process it."""
    dataset = pd.read_csv(filename)
    stop_words = model.Defaults.stop_words
    questions = dataset["Question"]
    wiki_ids = dataset["Wikipedia_ID"]
    dataset = []
    for question, wiki_id in zip(questions, wiki_ids):
        processed_query = query_processing(question, model, stop_words)
        dataset.append((processed_query, wiki_id))
    return dataset


def evaluate_models(question_file, model, rank_models):
    """Evaluate several models in one run"""
    dataset = load_dataset(question_file, model)
    tfidf_model = rank_models[1]
    bm_model = rank_models[0]
    # now calculate Mean Reciprocal Rank (MRR = 1/len(queries) * sum(1/rank_{true})
    # precision@k/R-precision at cutoff 1 (we only have one relevant document in corpus) ; k = 1
    sum_reciprocals_tf = 0
    sum_reciprocals_bm = 0
    number_of_queries = len(dataset)
    number_correct_tf = 0
    number_correct_bm = 0
    for query, correct_id in dataset:
        rank, correct = evaluate_okapi(query, correct_id, bm_model)
        sum_reciprocals_bm += 1/rank
        number_correct_bm += correct
        rank, correct = evaluate_tf_idf(query, correct_id, tfidf_model)
        sum_reciprocals_tf += 1/rank
        number_correct_tf += correct
    return (
        (1/number_of_queries * sum_reciprocals_bm, number_correct_bm/number_of_queries),
        (1/number_of_queries * sum_reciprocals_tf, number_correct_tf/number_of_queries)
    )


def evaluate_tf_idf(query, correct_id, tfidf_model):
    """Query TFIDFmodel and return reciprocal and boolean if rank == 1"""
    ranked_ids = tfidf_model.rank(query)
    rank = ranked_ids.index(correct_id) + 1
    reciprocal = 1/rank
    if rank == 1:  # we only check for r_1, so it's either 1 or precision at 1 is automatically 0
        correct = 1
    else:
        correct = 0
    return reciprocal, correct


def evaluate_okapi(query, correct_id, bm_model):
    """Query BM25 model and return reciprocal and boolean if rank == 1"""
    ranked_ids = bm_model.rank(query)
    rank = ranked_ids.index(correct_id) + 1  # add 1 because index starts at 0
    if rank == 1:
        correct = 1
    else:
        correct = 0
    reciprocal = 1/rank
    return reciprocal, correct


# TODO: Clean these two function up
def evaluate(datapath):
    """Run evaluation process."""
    model = spacy.load('en_core_web_sm')
    bm_model, inverted_index = load_utilities_for_bm()
    tfidf_model, inverted_index = load_utilities_for_tfidf()
    query_models = [bm_model, tfidf_model]
    # dev set
    (bm_mmr, bm_pr_k), (tf_mmr, tf_pr_k) = evaluate_models(join(datapath, "nq_dev.csv"), model, query_models)
    print("Okapi BM25 results on dev set:")
    print(f"MMR: {bm_mmr}, Precision@1: {tf_pr_k}")
    print("TFIDF with cosine similarity results on dev set:")
    print(f"MMR: {tf_mmr}, Precision@1: {tf_pr_k}")
    # see how good we are on the training set
    (bm_mmr, bm_pr_k), (tf_mmr, tf_pr_k) = evaluate_models(join(datapath, "nq_train.csv"), model, query_models)
    print("Okapi BM25 results on training set:")
    print(f"MMR: {bm_mmr}, Precision@1: {bm_pr_k}")
    print("TFIDF with cosine similarity results on training set:")
    print(f"MMR: {tf_mmr}, Precision@1: {tf_pr_k}")
    # TODO: Fine-tune models on development set


def evaluate_test(datapath):
    """Run evaluation process on test set."""
    model = spacy.load('en_core_web_sm')
    bm_model, inverted_index = load_utilities_for_bm()
    tfidf_model, inverted_index = load_utilities_for_tfidf()
    query_models = [bm_model, tfidf_model]
    (bm_mmr, bm_pr_k), (tf_mmr, tf_pr_k) = evaluate_models(join(datapath, "natural_questions_dev.csv"), model, query_models)
    print("Okapi BM25 results on test set:")
    print(f"MMR: {bm_mmr}, Precision@1: {bm_pr_k}")
    print("TFIDF with cosine similarity results on test set:")
    print(f"MMR: {tf_mmr}, Precision@1: {tf_pr_k}")


if __name__ == "__main__":
    evaluate_test(DATAPATH)
    evaluate(DATAPATH_PROCESSED)
