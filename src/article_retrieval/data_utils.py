#!/usr/bin/env python3


""" Several functions for data loading and processing; defined here for easier importing """

import json
import csv
import pickle
import re
from os.path import isfile

import pandas as pd
import spacy
from tqdm import tqdm


def read_index(filename):
    """Load a json file."""
    with open(filename) as f:
        return json.load(f)


def save_index(index, filename="inverted_index.json"):
    """Save a dictionary to disc as as json file."""
    with open(filename, "w") as f:
        json.dump(index, f)


def process_corpus(dataframe_filename, model):
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
    dataframe = []      # free memory
    counter = 0
    index2wikiid = {}
    for document, wiki_id in zip(corpus, wiki_ids):
        print("Processing document {}/{}".format(counter+1, len(corpus)))
        index2wikiid[counter] = wiki_id
        counter += 1
        processed_corpus.append(get_article_content(document, model, stop_words))
    # adapted from https://stackoverflow.com/questions/30711899/python-how-to-write-list-of-lists-to-file
    with open(f"{dataframe_filename.split('.csv')[0]}_lemmatized_stop_words_removed.csv", "w") as f:
        wr = csv.writer(f)
        for line in processed_corpus:
            wr.writerow(line)
    save_index(index2wikiid, f"{dataframe_filename.split('.csv')[0]}_index2key.json")
    return processed_corpus, index2wikiid


# TODO: Remove this once dataset preprocessing is complete
def lemmatize_sentence(sentence, model, stop_words):
    """Return lemmata of all words in sentence."""
    sentence = " ".join([word.lower() for word in sentence.split() if word not in stop_words])
    processed_sentence = model(sentence)
    lemmatas = [word.lemma_ for word in processed_sentence]
    return lemmatas


def get_article_content(article, model, stop_words):
    """Tokenize and lemmatize article."""
    # regex = re.compile(r"\n\n+")
    # article = re.sub(regex, "", article)
    article_paragraphs = article.split("\n\n")
    lemmatized_article = []
    for i, paragraph in enumerate(article_paragraphs):
        print("Lemmatizing paragraph {}/{}".format(i+1, len(article_paragraphs)), end='\r')
        lemmatized_paragraph = lemmatize_sentence(paragraph, model, stop_words)
        lemmatized_article.append(" ".join(lemmatized_paragraph))
    return "\n\n".join(lemmatized_article)


def load_corpus(filename):
    """Load preprocessed corpus from file if it exists, else create it."""
    split_filename = filename.split('.csv')[0]
    if isfile(f"{split_filename}_lemmatized_stop_words_removed.csv"):
        documents = []
        # adapted from https://docs.python.org/3.8/library/csv.html
        with open(f"{split_filename}_lemmatized_stop_words_removed.csv") as csvfile:
            docreader = csv.reader(csvfile)
            for row in docreader:
                documents.append(" ".join(row))
        index2key = read_index(f"{split_filename}_index2key.json")
    else:
        model = spacy.load('en_core_web_sm', exclude=["ner", "parser", "tagger", "tok2vec"])
        model.max_length = 10000000
        documents, index2key = process_corpus(filename, model)
    return documents, index2key
