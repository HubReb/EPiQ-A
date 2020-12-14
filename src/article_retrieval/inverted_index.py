#!/usr/bin/env python3

from collections import defaultdict

import spacy
import re

from data_utils import get_data, save_index


# TODO: Remove this once dataset preprocessing is complete
def lemmatize_sentence(sentence, model, stop_words):
    """ Return lemmatas of all words in sentence """
    sentence = " ".join([word.lower() for word in sentence.split() if word not in stop_words])
    processed_sentence = model(sentence)
    lemmatas = [word.lemma_ for word in processed_sentence]
    return lemmatas


def get_article_content(article, model, stop_words):
    regex = re.compile(r"\n\n+")
    article = re.sub(regex, "", article)
    lemmatized_article = lemmatize_sentence(article, model, stop_words)
    return lemmatized_article


def create_inverted_index(dataframe, model, stop_words):
    inverted_index = defaultdict(set)
    # TODO: find a faster way to iterate; iterrows() is very slow
    total = len(dataframe)
    for index, row in dataframe.iterrows():
        content = get_article_content(row["Text"], model, stop_words)
        wiki_key = row["Wikipedia_ID"]
        for word in content:
            inverted_index[word].add(wiki_key)
        if index % 100 == 0:
            print(index/total)
    final_index = {}
    for index, doc_ids in inverted_index.items():
        final_index[index] = list(doc_ids)
    return final_index


if __name__ == "__main__":
    model = spacy.load('en_core_web_sm')
    stop_words = model.Defaults.stop_words
    df = get_data("../data/nq_train_wiki_text.csv")
    inverted_index = create_inverted_index(df, model, stop_words)
    save_index(inverted_index)
