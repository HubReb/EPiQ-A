# -*- coding: utf-8 -*-

import re
import os
import csv
import sys
import json

from nltk.corpus import stopwords
from typing import List, Set, Dict
from question_parsing.question_parsing import Question
from question_parsing.question_parsing import parse_question

Stopwords = set(stopwords.words("english"))


def tokenize(tokens: str) -> List[str]:
    return [token.strip() for token in tokens.split()]


def is_valid_title(title: str) -> bool:
    """Checks whether to use title"""
    # Filter stopwords, we don't want articles named, for example, `a` here
    return any([token not in Stopwords for token in tokenize(title)])


def extract_titles(filename: str) -> Dict[str, str]:
    """
    Reads wikipedia article names from the raw source file that
    we extracted from the natural questions dataset.
    """
    titles = dict()
    # We need to temporarily remove the csv field size limit because some
    # articles are huge
    old_max_field_size = csv.field_size_limit()
    csv.field_size_limit(sys.maxsize)

    with open(filename) as wf:
        for _, link, text in csv.reader(wf):
            # Title is always in first line
            title = text.split('\n')[0].strip().lower()
            # Clean weird markup of the raw articles
            title = re.sub('- wikipedia', '', title).strip()
            if not is_valid_title(title):
                continue
            # Maybe some articles have the same title
            if title in titles:
                titles[title].append(link)
            else:
                titles[title] = [link]

    csv.field_size_limit(old_max_field_size)
    return titles


def make_inverted_index(titles: List[str]) -> Dict[str, Set[str]]:
    """
    Creates a simple inverted index mapping tokens to full titles in which
    the token appears.
    """
    inverted_index = dict()
    for title in titles:
        tokens = tokenize(title)
        for token in tokens:
            # We use sets during creation to avoid duplicate titles
            if token in inverted_index:
                inverted_index[token].add(title)
            else:
                inverted_index[token] = {title}
    
    # Since we want to save the index, we need jsonable datatypes
    return {key: tuple(value) for key, value in inverted_index.items()}


class ArticlesFromTitleMentions:
    """
    Provides functionality to retrieve wikipedia articles whose titles
    are mentioned in the question.
    """
    def __init__(self, filename: str = None):
        if filename.endswith('.json'):
            with open(filename) as saved_index:
                self.title2link,  self.title_index = json.load(saved_index)
        else:
            self.title2link = extract_titles(filename)
            titles = list(self.title2link.keys())
            self.title_index = make_inverted_index(titles)
    
    def save(self, filename: str = "article_title_index.json"):
        with open(filename, 'w') as sf:
            json.dump((self.title2link, self.title_index), sf)


    @staticmethod
    def match(question: Question, title: str):
        """Check if title is mentioned in question"""
        # Check exact string match
        return title in " ".join(question.original_terms)


    def get_articles_with_title_mentions(self, question: Question) -> List[str]:
        """Get articles whose title is mentioned in the given question"""
        relevant_titles = set()
        for term in question.original_terms:
            relevant_titles.update(self.title_index.get(term, set()))
            
        links = []
        for title in relevant_titles:
            if self.match(question, title):
                links.append(" ".join(self.title2link[title]))

        return links


if __name__ == '__main__':
    article_getter = ArticlesFromTitleMentions('../../data/article_retrieval/nq_dev_train_wiki_text.csv')
    with open("../question_parsing/test_questions.txt") as tf:
        for line in tf:
            question = line.strip()
            parse = parse_question(question, include_hyponyms=True, include_hypernyms=True)
            print(question)
            print(article_getter.get_articles_with_title_mentions(parse))
            print()
    article_getter.save()
