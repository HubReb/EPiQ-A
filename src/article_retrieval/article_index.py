import re
import csv
import sys
import json

from typing import List, Set, Dict
from question_parsing.question_parsing import Question, parse_question


def tokenize(tokens: str) -> List[str]:
    return [token.strip() for token in tokens.split()]


def extract_titles(filename: str) -> Dict[str, str]:
    titles = dict()
    old_max_filed_size = csv.field_size_limit()
    csv.field_size_limit(sys.maxsize)
    
    with open(filename) as wf:
        for _, link, text in csv.reader(wf):
            title = text.split('\n')[0].strip().lower()
            title = re.sub('- wikipedia', '', title).strip()
            titles[title] = link
    
    csv.field_size_limit(old_max_filed_size)
    return titles


def make_inverted_index(titles: List[str]) -> Dict[str, Set[str]]:
    inverted_index = dict()
    for title in titles:
        tokens = tokenize(title)
        for token in tokens:
            if token in inverted_index:
                inverted_index[token].add(title)
            else:
                inverted_index[token] = {title}
    
    return {key: tuple(value) for key, value in inverted_index.items()}


class ArticlesFromTitleMentions:
    def __init__(self, filename: str = None, autosave: bool = True):
        self.autosave = autosave
        if filename.endswith('.json'):
            with open(filename) as saved_index:
                self.title2link,  self.title_index = json.load(saved_index)
        else:
            self.title2link = extract_titles(filename)
            titles = list(self.title2link.keys())
            self.title_index = make_inverted_index(titles)
    
    
    #def __del__(self):
        #if self.autosave:
            #self.save()  # Enable automatic saving
    
    
    def save(self, filename: str = "article_title_index.json"):
        with open(filename, 'w') as sf:
            json.dump((self.title2link, self.title_index), sf)

    @staticmethod
    def match(question: Question, title: str):
        return title in " ".join(question.terms)


    def get_articles_with_title_mentions(self, question: Question) -> List[str]:
        relevant_titles = set()
        for term in question.terms:
            relevant_titles.update(self.title_index.get(term, set()))
        
        return [(title, self.title2link[title])
                for title in relevant_titles
                if self.match(question, title)]


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
