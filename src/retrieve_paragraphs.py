# -*- coding: utf-8 -*-

# The exponential input curve
import os
import csv
import sys
import nltk
import pickle
import numpy as np
from tqdm import tqdm
from typing import List
from gensim.summarization.bm25 import BM25
from question_parsing import Question


ARTICLESPATH = "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv"
BM25MODELFILE = "answer_extraction_bm25_{}_model.pkl"


class GetBestParagraphs:
    """
    A document ranker that utilizes the Okapi BM 25 weighting scheme.

    Attributes:
        level: {'paragraph', 'sentence'}
            whether articles are split on paragraph or sentence level
        bm25_model_file: str
            name of file to save BM25 model
        n_top_paragraphs: int
            number of paragraphs to retrieve
        max_context_size: int
            maximum size of context windows. Needed because transformers
            models have a maximum contexts size, while some paragraphs exceed
            this limit. We employ a sliding window approach to circumvent
            this problem.
        bm25: BM25
            the BM25 model used for ranking paragraphs
    """

    def __init__(
        self,
        level: str = "paragraph",
        n_top_paragraphs: int = 10,
        max_context_size: int = 400,
    ):
        self.level = level
        self.bm25_model_file = BM25MODELFILE.format(level)
        self.n_top_paragraphs = n_top_paragraphs
        self.max_context_size = max_context_size

        print("Loading paragraph data")
        self.bm25, self.paragraphs, self.wiki_index2paragraph_index = self.load_data()

    def iter_paragraphs(self, text: str):
        """
        Splits given text according to splitting strategy
        (paragraphs/sentences). Iters through received paragraphs.

        Arguments:
            text - Article text to split
        Raises:
            ValueError - if specified splitting strategy is invalid
        """
        if self.level == "paragraph":
            yield from text.split("\n\n")
        elif self.level == "sentence":
            yield from nltk.sent_tokenize(text)
        else:
            raise ValueError("Unknown paragraph level: {}".format(self.level))

    def load_data(self):
        """
        Loads paragraphs, wikipedia identifier to paragraph index mapping, and
        trains BM25 model
        """
        print("Paragraph retrieval model - Loading paragraphs")
        articles_csv_path = ARTICLESPATH
        paragraphs = []
        current_paragraph_id = 0
        wiki_index2paragraph_index = dict()

        csv.field_size_limit(sys.maxsize)

        with open(articles_csv_path) as af:
            for _, key, text in tqdm(csv.reader(af, delimiter=",")):
                wiki_index2paragraph_index[key] = []

                for paragraph in self.iter_paragraphs(text):
                    # Simplistic tokenisation
                    paragraph = paragraph.strip().split()
                    if not paragraph:
                        continue

                    wiki_index2paragraph_index[key].append(current_paragraph_id)
                    current_paragraph_id += 1

                    paragraphs.append(paragraph)

        print("Paragraph retrieval model - Loading BM25 model")
        if os.path.isfile(self.bm25_model_file):
            with open(self.bm25_model_file, "rb") as bm25_file:
                bm25 = pickle.load(bm25_file)
        else:
            print("Training new model. This may take some time.")
            bm25 = BM25(paragraphs)
            with open(self.bm25_model_file, "wb") as bm25_file:
                pickle.dump(bm25, bm25_file)

        return bm25, paragraphs, wiki_index2paragraph_index

    def partition_paragraph(self, paragraph: List[str]) -> List[str]:
        """
        Creates context containing at most `self.max_context_size` tokens.
        Needed for sliding window approach of answer extraction from
        paragraphs.

        Arguments:
            paragraph - (Tokenised) paragraph
        Returns:
            contexts - List of context windows
        """
        # Don't create windows if paragraph is shorter than
        # `self.max_context_size`
        if len(paragraph) <= self.max_context_size:
            return [" ".join(paragraph)]

        # Create context windows from paragraph
        else:
            start = 0
            contexts = []
            while start < len(paragraph):
                stop = start + self.max_context_size
                stop = min(stop, len(paragraph))

                window = paragraph[max(0, start - 20) : stop]
                context = " ".join(window)
                contexts.append(context)

                start = stop
            return [context for context in contexts if context.strip()]

    def prepare_contexts(self, paragraphs: List[List[str]]) -> List[str]:
        """
        Creates context windows containing at most `self.max_context_size`
        tokens from all multiple contexts.
        Needed for sliding window approach of answer extraction from
        paragraphs.

        Arguments:
            paragraphs - (Tokenised) paragraphs
        Returns:
            contexts - List of context windows
        """
        contexts = []
        for paragraph in paragraphs:
            contexts.extend(self.partition_paragraph(paragraph))
        return contexts

    def get_best_paragraphs(
        self, question: Question, article_ids: List[str] = None
    ) -> List[str]:
        """
        Retrieve best matching paragraphs (by BM25 model) given the parsed
        question. If `article_ids` is given, only consider paragraphs
        from the specified articles.

        Arguments:
            question - The parsed question
            article_ids (optional) - Wikipedia Links of retrieved articles

        Returns:
            contexts - Context windows from top scoring paragraphs
        """
        if article_ids is not None:
            paragraph_ids = []
            for article_id in article_ids:
                paragraph_ids.extend(self.wiki_index2paragraph_index[article_id])

            # Extract top-scoring paragraphs
            scores = [
                self.bm25.get_score(question.original_terms, paragraph_id)
                for paragraph_id in paragraph_ids
            ]

        else:
            scores = self.bm25.get_scores(question.original_terms)

        top_scoring_paragraph_indices = np.argsort(scores)[::-1][
            : self.n_top_paragraphs
        ]
        top_scoring_paragraphs = [
            self.paragraphs[index] for index in top_scoring_paragraph_indices
        ]

        return self.prepare_contexts(top_scoring_paragraphs)
