#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Question answering pipeline class definition """

import os
from time import time
from typing import List
from transformers import pipeline
from helper_functions import load_csv
from retrieve_paragraphs import GetBestParagraphs
from article_retrieval.gensim_bm25 import Okapi25
from article_retrieval.query_index import query_index
from article_retrieval.config import (
    BM25_MODEL,
    TFIDF_MODEL,
    INVERTED_INDEX,
    TITLE_INDEX
)
from article_retrieval.data_utils import (
    load_utilities_for_bm,
    load_utilities_for_tfidf
)
from article_retrieval.article_index import ArticlesFromTitleMentions
from question_parsing import parse_question, Question


def check_all_files_exist():
    """Checks whether the data was successfully set up and models are trained"""
    assert os.path.isfile(INVERTED_INDEX)
    assert os.path.isfile(TITLE_INDEX)
    assert os.path.isfile("../data/article_retrieval/nq_dev_train_wiki_text_merged.csv")
    assert os.path.isfile(BM25_MODEL)
    assert os.path.isfile(TFIDF_MODEL)

    return None


class CombinedModel:
    """ Pipeline of all modules of the question answering system for end-to-end usage """
    def __init__(
        self,
        article_retrieval_model: str,
        model_filename: str,
        article_retrieval_use_index: bool = True,
        retrieve_articles: bool = True,
        num_articles: int = 10,
        answer_extraction_model_name: str = None,
        paragraph_level: str = "paragraph",
        n_top_paragraphs: int = 10,
        max_context_size: int = 400,
    ):
        check_all_files_exist()
        self.article_retrieval_use_index = article_retrieval_use_index
        self.retrieve_articles = retrieve_articles
        self.max_articles = num_articles

        self.question_parser = parse_question

        print("Loading article retrieval model")
        if article_retrieval_model == "bm":
            bm_model, bm_inverted_index = load_utilities_for_bm(
                model_filename, "article_retrieval/inverted_index.json"
            )
            self.article_model = bm_model
            self.inverted_index = bm_inverted_index
        elif article_retrieval_model == "tfidf":
            tfidf_model, tfidf_inverted_index = load_utilities_for_tfidf(
                model_filename, "article_retrieval/inverted_index.json"
            )
            self.article_model = tfidf_model
            self.inverted_index = tfidf_inverted_index
        else:
            raise ValueError("Unknown model: {}".format(article_retrieval_model))

        print("Loading article title index model")
        self.get_articles_from_titles = ArticlesFromTitleMentions(
            "article_retrieval/article_title_index.json"
        )
        self.get_articles_from_titles = (
            self.get_articles_from_titles.get_articles_with_title_mentions
        )

        print("Loading paragraph retrieval model")
        self.paragraph_retrieval_model = GetBestParagraphs(
            paragraph_level, n_top_paragraphs, max_context_size
        )
        self.paragraph_retrieval_model = (
            self.paragraph_retrieval_model.get_best_paragraphs
        )

        print("Loading (pretrained) Q/A model")
        self.answer_extraction_model = pipeline(
            "question-answering", model=answer_extraction_model_name
        )

    def get_answer(self, query: str):
        """ Apply pipeline to a single query and return answer """
        parsed_query = self.question_parser(question=query)
        ranked_ids = None

        if self.retrieve_articles:
            ranked_ids = self.get_ranked_article_ids(parsed_query)

        top_contexts = self.paragraph_retrieval_model(parsed_query, ranked_ids)
        answer = self.get_answer_from_contexts(parsed_query, top_contexts)

        return answer

    def get_ranked_article_ids(self, question: Question) -> List[str]:
        """ Apply article_retrieval module to score and rank articles in regards to question."""
        if self.article_retrieval_use_index:
            docs, _ = query_index(question, self.inverted_index, self.article_model)
            ranked_ids = self.article_model.rank_docs(question, docs)
        else:
            ranked_ids_ = self.article_model.rank(question)
            ranked_ids_ = ranked_ids_[: self.max_articles]
            ranked_ids = []
            for links in ranked_ids_:
                ranked_ids.append(" ".join(links))

        ranked_ids.extend(self.get_articles_from_titles(question))
        ranked_ids = list(set(ranked_ids))
        return ranked_ids

    def get_answer_from_contexts(self, question: Question, contexts: List[str]) -> str:
        """
        Use pretrained transformers model to extract answer with highest
        confidence from given contexts

        Arguments:
            question - The parsed question
            contexts - Possible contexts to extract answer from
        Returns:
            answer - Highest scoring anwer span from contexts
        """
        question = " ".join(question.original_terms)

        answers = []
        for context in contexts:
            answer = self.answer_extraction_model(question=question, context=context)
            answers.append(answer)

        answers = sorted(answers, key=lambda a: a["score"], reverse=True)
        best_answer = next(iter(answers))
        return best_answer["answer"]


if __name__ == "__main__":
    model = CombinedModel(
        "tfidf", TFIDF_MODEL, article_retrieval_use_index=False, retrieve_articles=False
    )
    # model.get_answer("who had most wins in nfl")
    # print()
    # model = CombinedModel("bm", "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv", BM25_MODEL)
    # model.get_answer("who had most wins in nfl")

    question_dev_dataframe = load_csv("../data/natural_questions_train.csv")

    print("Predicting answers...")
    for i, row in question_dev_dataframe.iterrows():
        starttime = time()
        question = row["Question"]
        answer = model.get_answer(question)
        endtime = time()

        print("Question:", question)
        print("Answer:", answer)
        print("Needed:", endtime - starttime, "seconds")
        print()
