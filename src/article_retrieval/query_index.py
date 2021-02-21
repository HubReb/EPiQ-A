#!/usr/bin/env python3

"""Define the functions to query the invered index"""

from collections import defaultdict
import pickle
from typing import List, Union

import spacy


from article_retrieval.data_utils import read_index
from article_retrieval.article_index import ArticlesFromTitleMentions
from article_retrieval.tf_idfs import TFIDFmodel
from article_retrieval.gensim_bm25 import Okapi25


def query_processing(
    query: str, model: spacy.language.Language, stop_words: List[str]
) -> List[str]:
    """ Remove stop words from query. tokenize and lemmatize it"""
    query = " ".join([word.lower() for word in query.split() if word not in stop_words])
    processed_query = model(query)
    return [word.lemma_ for word in processed_query]


def query_index(
    query: str,
    inverted_index: dict,
    model: Union[Okapi25, TFIDFmodel],
    processing: bool = False,
    must_have: bool = None,
) -> (List[int], List[str]):
    """
    Retrieve doc indices relevant to query from inverted index

    Process the query to remove stop words, lemmatize and tokenize it.
    Then query the inverted index to retrieve the documents which contain
    words of the query.

    Arguments:
        query - query to retrieve documents for
        inverted_index - dictionary from words to the document indices that contain them
        model - ranking model for getting model's stop word list
        processing - boolean to determine if preprocessing is necessary
        must_have - list of terms that must be present in the document to be drawn from index

    Returns:
        best_doc_guesses - indices of documents that contain words of the query
        query - processed query

    Raises:
        ValueError if must_have is specifed but an empty list
    """
    stop_words = model.stop_words
    document_ids = defaultdict(list)
    if processing:
        query_tokens = query_processing(query, spacy.load("en_core_web_sm"), stop_words)
    else:
        query_tokens = query.terms
    if must_have:
        if must_have == []:
            raise ValueError("We cannot query the index with an empty list!")
    for word in query_tokens:
        if must_have and word not in must_have:
            continue
        if word in inverted_index.keys():
            document_ids[word].extend(inverted_index[word])
    docs_to_query_words_counter = defaultdict(int)
    for word, ids in document_ids.items():
        for doc_id in ids:
            docs_to_query_words_counter[doc_id] += 1
    best_counter_doc_guesses = sorted(
        [d_id for d_id in docs_to_query_words_counter.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    best_doc_guesses = [doc_id for doc_id, counter in best_counter_doc_guesses]
    return best_doc_guesses, query_tokens


if __name__ == "__main__":
    from question_parsing import parse_question

    index = read_index("inverted_index.json")
    with open("tfidfmodel.pkl", "rb") as f:
        tfidf_model = pickle.load(f)
    docs, example_query = query_index(
        "Who was George Bush?", index, tfidf_model, processing=True
    )
    print(tfidf_model.rank_docs(docs, example_query, evaluate_component=True)[:10])
    parse = parse_question("Who was George Bush?")
    docs, query = query_index(parse, index, tfidf_model)
    print(tfidf_model.rank_docs(docs, query, evaluate_component=True)[:10])
