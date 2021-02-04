#!/usr/bin/env python3

""" Evaluation script for baselines """

from os.path import join
import pickle
import json

import spacy
import pandas as pd

from article_retrieval.query_index import query_index, query_processing
from article_retrieval import gensim_bm25
from article_retrieval.gensim_bm25 import Okapi25
from article_retrieval.config import DATAPATH_PROCESSED, DATAPATH


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

def load_from_pickle(filename):
    """Load dataset from pickled file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_from_json(filename):
    """Load dictionary from json file."""
    with open(filename) as f:
        return json.load(f)


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
    p_at_ks_tf = {}
    p_at_ks_bm = {}
    for i in range(2, 11):
        p_at_ks_tf[i+1] = 0
        p_at_ks_bm[i+1] = 0
    for idx, (query, correct_id) in enumerate(dataset):
        print("Evaluating query {}/{}".format(idx+1, len(dataset)), end='\r')
        #reciprocal, correct, correct_rank = evaluate_okapi(query, correct_id, bm_model)
        #for rank in p_at_ks_bm.keys():
            #if rank <= correct_rank:
                #p_at_ks_bm[rank] += 1/rank
        #sum_reciprocals_bm += reciprocal
        #number_correct_bm += correct
        
        reciprocal, correct, correct_rank = evaluate_tf_idf(query, correct_id, tfidf_model)
        for rank in p_at_ks_tf.keys():
            if rank >= correct_rank:
                p_at_ks_tf[rank] += 1/rank
        sum_reciprocals_tf += reciprocal
        number_correct_tf += correct

    print()
    print("number", number_of_queries, "sum_reciprocals_bm", sum_reciprocals_bm, "number_correct: ", number_correct_bm)
    print("number", number_of_queries, "sum_reciprocals_tf", sum_reciprocals_tf, "number_correct: ", number_correct_tf)
    for rank, n_correct in p_at_ks_bm.items():
        p_at_ks_bm[rank] = n_correct/number_of_queries
    for rank, n_correct in p_at_ks_tf.items():
        p_at_ks_tf[rank] = n_correct/number_of_queries
    return (
        (1/number_of_queries * sum_reciprocals_bm, number_correct_bm/number_of_queries, p_at_ks_bm),
        (1/number_of_queries * sum_reciprocals_tf, number_correct_tf/number_of_queries, p_at_ks_tf)
    )


def evaluate_tf_idf(query, correct_id, tfidf_model):
    """Query TFIDFmodel and return reciprocal and boolean if rank == 1"""
    ranked_ids = tfidf_model.rank(query=query, evaluate_component=True)
    if all([isinstance(link, str) for link in ranked_ids]):  # Model returns single link
        rank = ranked_ids.index(correct_id) + 1
    else:  # Model retrieves merged articles, returns multiple links
        ranks = [rank+1 for rank, links in enumerate(ranked_ids)
                 if correct_id in links]
        # assert len(ranks) == 1
        rank = ranks[0]

    reciprocal = 1/rank
    if rank == 1:  # we only check for r_1, so it's either 1 or precision at 1 is automatically 0
        correct = 1
    else:
        correct = 0
    return reciprocal, correct, rank


def evaluate_okapi(query, correct_id, bm_model):
    """Query BM25 model and return reciprocal and boolean if rank == 1"""
    ranked_ids = bm_model.rank(query=query, evaluate_component=True)
    rank = ranked_ids.index(correct_id) + 1  # add 1 because index starts at 0
    if rank == 1:
        correct = 1
    else:
        correct = 0
    reciprocal = 1/rank
    return reciprocal, correct, rank


# TODO: Clean these two function up
def evaluate(datapath):
    """Run evaluation process."""
    model = spacy.load('en_core_web_sm')
    bm_model, inverted_index = load_utilities_for_bm()
    tfidf_model, inverted_index = load_utilities_for_tfidf()
    query_models = [bm_model, tfidf_model]
    # dev set
    (bm_mmr, bm_pr_k, bm_p_ks), (tf_mmr, tf_pr_k, tf_pks) = evaluate_models(join(datapath, "nq_dev.csv"), model, query_models)
    print("Okapi BM25 results on dev set:")
    print(f"MMR: {bm_mmr}, Precision@1: {bm_pr_k}")
    print("TFIDF with cosine similarity results on dev set:")
    print(f"MMR: {tf_mmr}, Precision@1: {tf_pr_k}")
    print("Precision@k")
    for rank, p in p_at_k.items():
        print(f"k = {rank}: {p}")
    # see how good we are on the training set
    (bm_mmr, bm_pr_k, bm_p_at_k), (tf_mmr, tf_pr_k, tf_p_at_k) = evaluate_models(join(datapath, "nq_train.csv"), model, query_models)
    print("Okapi BM25 results on training set:")
    print(f"MMR: {bm_mmr}, Precision@1: {bm_pr_k}")
    print("TFIDF with cosine similarity results on training set:")
    print(f"MMR: {tf_mmr}, Precision@1: {tf_pr_k}")
    for rank, p in bm_p_at_k.items():
        print(f"k = {rank}: {p}")
    for rank, p in tf_p_at_k.items():
        print(f"k = {rank}: {p}")
    # TODO: Fine-tune models on development set


def evaluate_test(datapath):
    """Run evaluation process on test set."""
    model = spacy.load('en_core_web_sm')
    bm_model, inverted_index = load_utilities_for_bm()
    tfidf_model, inverted_index = load_utilities_for_tfidf()
    query_models = [bm_model, tfidf_model]
    (bm_mmr, bm_pr_k, bm_p_ks), (tf_mmr, tf_pr_k, tf_pks) = evaluate_models(join(datapath, "natural_questions_dev.csv"), model, query_models)
    print("Okapi BM25 results on dev set:")
    print(f"MMR: {bm_mmr}, Precision@1: {bm_pr_k}")
    print("TFIDF with cosine similarity results on dev set:")
    print(f"MMR: {tf_mmr}, Precision@1: {tf_pr_k}")
    print("Precision@k")
    for rank, p in bm_p_ks.items():
        print(f"k = {rank}: {p}")
    for rank, p in tf_pks.items():
        print(f"k = {rank}: {p}")
    print("Okapi BM25 results on test set:")
    print(f"MMR: {bm_mmr}, Precision@1: {bm_pr_k}")
    print("TFIDF with cosine similarity results on test set:")
    print(f"MMR: {tf_mmr}, Precision@1: {tf_pr_k}")


if __name__ == "__main__":
    evaluate_test(DATAPATH)
    evaluate(DATAPATH_PROCESSED)
