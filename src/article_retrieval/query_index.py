#!/usr/bin/env python3

"""Define the functions to query the invered index"""

from collections import defaultdict
import pickle

import spacy

from article_retrieval.data_utils import read_index


def query_processing(query, model, stop_words):
    """ Remove stop words from query. tokenize and lemmatize it"""
    query = " ".join([word.lower() for word in query.split() if word not in stop_words])
    processed_query = model(query)
    return [word.lemma_ for word in processed_query]


def query_index(query, inverted_index, model):
    """
    Retrieve doc indices relevant to query from inverted index

    Process the query to remove stop words, lemmatize and tokenize it.
    Then query the inverted index to retrieve the documents which contain
    words of the query.

    Arguments:
        query - query to retrieve documents for
        inverted_index - dictionary from words to the document indices that contain them

    Returns:
        best_doc_guesses - indices of documents that contain words of the query
        query - processed query
        model - spacy model for lemmatization and stop word list
    """
    stop_words = model.Defaults.stop_words
    document_ids = defaultdict(list)
    query = query_processing(query, model, stop_words)
    for word in query:
        if word in inverted_index.keys():
            document_ids[word].extend(inverted_index[word])
    docs_to_query_words_counter = defaultdict(int)
    for word, ids in document_ids.items():
        for doc_id in ids:
            docs_to_query_words_counter[doc_id] += 1
    best_counter_doc_guesses = sorted(
            [d_id for d_id in docs_to_query_words_counter.items()], key=lambda x: x[1], reverse=True
        )
    best_doc_guesses = [doc_id for doc_id, counter in best_counter_doc_guesses]
    return best_doc_guesses, query


if __name__ == "__main__":
    model = spacy.load('en_core_web_sm')
    print("loaded spacy")
    index = read_index("inverted_index.json")
    with open("tfidfmodel.pkl", "rb") as f:
        tfidf_model = pickle.load(f)
    docs, query = query_index("Who was George Bush?", index, model)
    print(
        tfidf_model.rank_docs(
            docs,
            query
        )
    )
