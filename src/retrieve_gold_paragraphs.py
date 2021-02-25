# -*- coding: utf-8 -*-

# The exponential input curve
import csv
import sys
import nltk
from tqdm import tqdm
from typing import List


ARTICLESPATH = "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv"


class GetGoldParagraphs:

    def __init__(
        self,
        answer: str,
        level: str = "paragraph",
        max_context_size: int = 400
    ):
        self.answer = answer
        self.level = level
        self.max_context_size = max_context_size

        print("Loading gold paragraph data")
        self.paragraphs, self.wiki_index2paragraph_index = self.load_data()

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
        Loads paragraphs, wikipedia identifier to paragraph index mapping
        """
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

        return paragraphs, wiki_index2paragraph_index

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

    def get_gold_paragraphs(self) -> List[str]:
        """
        Retrieve gold paragraphs (from data)

        Returns:
            contexts - Context windows from top scoring paragraphs
        """
        paragraphs = [paragraph for paragraph in self.paragraphs if self.answer.lower() in " ".join(paragraph).lower()]

        return self.prepare_contexts(paragraphs)
