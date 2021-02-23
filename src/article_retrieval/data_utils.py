#!/usr/bin/env python3


""" Several functions for data loading and processing; defined here for easier importing """

import json
import csv
import pickle
from os.path import isfile
from typing import List

import pandas as pd
import spacy



def read_index(filename: str):
    """Load a json file."""
    with open(filename) as f:
        return json.load(f)


def save_index(index: dict, filename: str = "inverted_index.json"):
    """Save a dictionary to disc as as json file."""
    with open(filename, "w") as f:
        json.dump(index, f)


def load_utilities_for_bm(
    bm_file: str = "okapibm25.pkl", index_file: str = "inverted_index.json"
):
    """Load Okapi25 model and inverted index from their respective files."""
    bm25 = load_from_pickle(bm_file)
    inverted_index = read_index(index_file)
    return bm25, inverted_index


def load_utilities_for_tfidf(
    model_file: str = "tfidfmodel.pkl", index_file: str = "inverted_index.json"
):
    """Load TFIDFmodel model and inverted index from their respective files."""
    tfidf_model = load_from_pickle(model_file)
    inverted_index = read_index(index_file)
    return tfidf_model, inverted_index


def load_from_pickle(filename: str):
    """Load model from pickled file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def process_corpus(
    dataframe_filename: str, model: spacy.language.Language
) -> (List[List[str]], dict):
    """
    Process data, return corpus and id:wikipedia article identifier.

    Finish pre-processing of data, create dataframe row to article id
    dictionary and save corpus to disk.

    Arguments:
        dataframe_filename: filename of the csv-file containing the corpus
        model - spacy model to process data
    Returns:
        processed_corpus: (list of lists of strings) - lemmatized corpus
        with stop words removed
        index2wikiid: dictionary that maps indices of processed_corpus to
        the wikipedia articles' identifiers
    """
    dataframe = pd.read_csv(dataframe_filename)
    corpus = dataframe["Text"].values
    stop_words = model.Defaults.stop_words
    wiki_ids = dataframe["Wikipedia_ID"].values
    processed_corpus = []
    dataframe = []  # free memory
    counter = 0
    index2wikiid = {}
    for document, wiki_id in zip(corpus, wiki_ids):
        if counter % 1000 == 0:
            print("Processing document {}/{}".format(counter + 1, len(corpus)))
        index2wikiid[counter] = wiki_id
        counter += 1
        processed_corpus.append(get_article_content(document, model, stop_words))
    # adapted from https://stackoverflow.com/questions/30711899/python-how-to-write-list-of-lists-to-file
    with open(
        f"{dataframe_filename.split('.csv')[0]}_lemmatized_stop_words_removed.csv", "w"
    ) as f:
        wr = csv.writer(f)
        for line in processed_corpus:
            wr.writerow(line)
    save_index(index2wikiid, f"{dataframe_filename.split('.csv')[0]}_index2key.json")
    return processed_corpus, index2wikiid


def lemmatize_sentence(
    sentence: str, model: spacy.language.Language, stop_words: List[str]
) -> List[str]:
    """Return lemmata of all words in sentence."""
    sentence = " ".join(
        [word.lower() for word in sentence.split() if word not in stop_words]
    )
    processed_sentence = model(sentence)
    lemmatas = [word.lemma_ for word in processed_sentence]
    return lemmatas


def get_article_content(
    article: str, model: spacy.language.Language, stop_words: List[str]
) -> List[List[str]]:
    """Tokenize and lemmatize article."""
    article_paragraphs = article.split("\n\n")
    lemmatized_article = []
    for paragraph in article_paragraphs:
        lemmatized_paragraph = lemmatize_sentence(paragraph, model, stop_words)
        lemmatized_article.extend(lemmatized_paragraph)
    return lemmatized_article


def load_corpus(filename: str) -> (List[str], dict):
    """Load preprocessed corpus from file if it exists, else create it."""
    split_filename = filename.split(".csv")[0]
    if isfile(f"{split_filename}_lemmatized_stop_words_removed.csv"):
        documents = []
        # adapted from https://docs.python.org/3.8/library/csv.html
        with open(f"{split_filename}_lemmatized_stop_words_removed.csv") as csvfile:
            docreader = csv.reader(csvfile)
            for row in docreader:
                documents.append(" ".join(row))
        index2key = read_index(f"{split_filename}_index2key.json")
    else:
        model = spacy.load(
            "en_core_web_sm", exclude=["ner", "parser", "tagger", "tok2vec"]
        )
        model.max_length = 10000000
        documents, index2key = process_corpus(filename, model)
    return documents, index2key
