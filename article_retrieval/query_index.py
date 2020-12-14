#!/usr/bin/env python3

from collections import defaultdict
import json
import pickle

import spacy
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import read_index, get_data


def query_processing(query, model, stop_words):
    query = " ".join([word.lower() for word in query.split() if word not in stop_words])
    processed_query = model(query)
    return [word.lemma_ for word in processed_query]


def query_index(query, model, inverted_index, stop_words):
    processed_query = query_processing(query, model)
    document_ids = defaultdict(list)
    for word in processed_query:
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
    return best_doc_guesses, processed_query


def rank_docs(docs, query, document_vectors, vectorizer, model, dataframe):
    similarities = []
    query_vector = vectorizer.transform(query)
    for doc in docs:
        document_vector = document_vectors[doc]
        similarities.append((doc, cosine_similarity(query_vector, document_vector)))
    ranked_sims = sorted(similarities, key=lambda x: x[1], reverse=True)[:10]
    ranked_docs = [doc_sim[0] for doc_sim in ranked_sims]
    return [dataframe[doc] for doc in ranked_docs]


model = spacy.load('en_core_web_sm')
stop_words = model.Defaults.stop_words
index = read_index("inverted_index.json")
df = get_data("../data/nq_train_wiki_text.csv")
with open("document_vectors.json") as f:
    doc_vectors = json.load(f)
with open("tf_idf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
docs, query = query_index("Who was George Bush?", model, index, stop_words)
print(
    rank_docs(
        docs,
        query,
        doc_vectors,
        vectorizer,
        model,
        df
    )
)
