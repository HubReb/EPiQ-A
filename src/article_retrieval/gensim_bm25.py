#!/usr/bin/env python3

"""Train document ranking model with Okapi BM 25 weights."""

from collections import namedtuple
from os.path import isfile
from typing import List

import spacy

from gensim.summarization.bm25 import BM25
from article_retrieval.data_utils import load_corpus, load_from_pickle


class Okapi25:
    """A document ranker that utilizes the Okapi BM 25 weighting scheme.

    The class uses gensim.summarization.bm25 module to build the weights.
    Forther information on the BM 25 weighting process can be found in
    gensim's module information. Note that the class parameters are
    given to gensim's module, here only a minimal documentation is provided
    as gensim's module documentation on them gives a thorough explanation.
    The default parameters are currently taken from gensim's documentation.

    Attributes:
        k: float
            the k value in weighting scheme that influences the term frequency
            saturation
        b: float
            the b value in weighting scheme that influences the average document
            length
        epsilon: float
            This is the floor value for idf to prevent negative idf values
            (if > 0.5 of all documents contain a word, idf is negative);
            epsilon > 0 increases how rare a term has to be to receive an extra
            score as negative idfs are replaced with epsilon * average_idf
        self.model: gensim.summarization.bm25.BM25 object
        self.index2wikiid: dict
            Maps self.models.doc_freqs indices to the wikipedia article
            identifiers.

    Methods:
        load(self, filename)
            Load trained model from file.

            filename: name of the file to load the model from
            Returns:
                loaded model

        fit(self, dataframe_filename):
            Fit gensim's BM25 model to data.

            dataframe_filename - name of the csv file the dataset is stored in

        rank(self, query_tuple, query, evaluate_component):
            Rank all documents in self.models.doc_freqs against a query.

            query_tuple (default: None)- named tuple of type Question
            query  (default: None)- processed query
            evaluate_component (default: False)- boolean to determine if we are
                only evaluating the retrieval component

            Returns:
            ranked list of wikipedia article identifiers

            Raises:
                TypeError if both query_tuple and query or evaluate_component are given
                TypeError if neither query_tuple nor query and evaluate_component are given

        rank_docs(self, query, docs, evaluate_component):
            Rank a subset of the docs in self.doc_freqs against a query.

            query - processed query or named tuple of type Question
            docs - indices of documents to calculate score for
            evaluate_component (default: False) - boolean to determine if we are
                only evaluating the retrieval component

            Returns:
                ranked list of wikipedia article identifiers

            Raises:
            TypeError if query is tuple and evaluate_component is True
    """

    def __init__(
        self,
        k: int = 1.5,
        b: int = 0.75,
        epsilon: int = 0.25,
        filename: str = "okapibm25.pkl",
    ):
        """
        Parameters:
            k: float
                The value to set self.k to.
            b: float
                The value to set self.b to.
            epsilon: float
                The value to set self.epsilon to.
            filename: string
                The name of the file the trained Okapi25 model is stored in.
                The model is loaded if the file exists.
        """

        self.k = k
        self.b = b
        self.epsilon = epsilon
        self.index2wikiid = {}
        self.stop_words = spacy.load("en_core_web_sm").Defaults.stop_words
        if isfile(filename):
            self.model = load_from_pickle(filename)

    def fit(self, dataframe_filename: str):
        """Fit gensim's BM25 model to data."""
        processed_corpus, self.index2wikiid = load_corpus(dataframe_filename)
        self.index2wikiid = {
            key: links.split(" ") for key, links in self.index2wikiid.items()
        }
        self.model = BM25(processed_corpus, self.k, self.b, self.epsilon)

    def rank(
        self,
        query_tuple: namedtuple = None,
        query: List[str] = None,
        evaluate_component: bool = False,
    ) -> List[str]:
        """
        Rank all documents in self.models.doc_freqs against a query.

        All documents in the corpus the BM25 model was trained on are ranked in
        decreasing order against a query. The self.doc_freqs indices are mapped
        to the wikipedia article identifiers before the ranked list is returned.
        The query must either have already been tokenized, lemmatized and have had
        its stop words removed be a namedTuple as created by the question_parsing.

        Arguments:
            query_tuple (default: None) - namedTuple of Question as defined by
                question_parsing component, cannot be combined with query or
                evaluate_component flag
            query (default: None) - processed query that must be lemmatized,
                 tokenized and have had stop words removed, must be set with
                 evaluate_component
            evaluate_component (default: False) - boolean value determining if
                the evaluation setup is executed

        Returns:
            ranked list of wikipedia article identifiers corresponding to the indices
            of all documents in the dataset
        Raises:
            TypeError if both query_tuple and query or evaluate_component are given
            TypeError if neither query_tuple nor query and evaluate_component are given
        """
        if query_tuple:
            if query or evaluate_component:
                raise TypeError(
                    "namedTuple Question is mutuably exlusive with evaluation_component"
                    "flag and processed query string!"
                )
        else:
            if not (query and evaluate_component):
                raise TypeError(
                    "Evaluation setup requires both processed question and evaluate_component flag"
                )

        if evaluate_component:
            query = " ".join(query)
        else:
            query = query_tuple.terms
        scores = self.model.get_scores(query)
        scores_index_tuples = [(index, score) for index, score in enumerate(scores)]
        ranked_scores = sorted(scores_index_tuples, key=lambda x: x[1], reverse=True)
        return [self.index2wikiid[str(score_tuple[0])] for score_tuple in ranked_scores]

    def rank_docs(
        self,
        query: List[str],
        docs: List[int],
        evaluate_component: bool = False,
        max_docs: int = 10,
    ) -> List[str]:
        """
        Rank a subset of the docs in self.doc_freqs against a query.

        The documents contained in docs are ranked against the query. All documents
        must be contained in the corpus the model was trained on. The indices in
        docs are mapped to the wikipedia article identifiers before the ranked
        list is returned. The query must have already been tokenized, lemmatized
        and have had its stop words removed.


        Arguments:
            query - processed query, either a tokenized list or a named tuple (Question)
            docs - indices of documents to calculate score for
            evaluate_component (default: False) - boolean to determine which query format is given
            max_docs (default: 10) - maximum number of (merged) articles to return

        Returns:
            A list of wikipedia article identifiers, sorted in decreasing
            similarity.

        Raises:
            TypeError if query is tuple and evaluate_component is True
        """

        if evaluate_component and not isinstance(query, list):
            raise TypeError(
                "namedTuple Question is mutuably exlusive with evaluation_component"
                "flag and processed query string!"
            )
        if evaluate_component:
            query = " ".join(query)
        else:
            query = " ".join(query.terms)
        scores_index_tuples = [
            (index, self.model.get_score(query, index)) for index in docs
        ]
        ranked_scores = list(
            sorted(scores_index_tuples, key=lambda x: x[1], reverse=True)
        )
        return [
            " ".join(self.index2wikiid[str(score_tuple[0])])
            for score_tuple in ranked_scores[:max_docs]
        ]
