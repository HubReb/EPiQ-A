#!/usr/bin/env python3


""" Define model to rank documents relative to a query with cosine similarity of TFIDF vectors"""

from typing import List, Union
from collections import namedtuple

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD


from article_retrieval.data_utils import load_corpus


class TFIDFmodel:
    """
    A document ranker that utilizes cosine similarity between TF-IDF vectors.

    The class uses sklearn.feature_extraction.text.TfidfVectorizer to create
    TF-IDF vectors for each document in the dataset. The model can then be
    used to rank either selected or all documents in the dataset against
    a query via cosine similarity between document TFIDF vector and the
    TFIDF vector of the query.

    Attributes:
        index2key: dictionary
            mapping from the index of a document in the dataset to the
            corresponding wikipedia article identifier
        doc_vecs:
            sparse matrix holding tf-idf feature vectors for all wikipedia
            articles
        truncated_doc_vecs: np.ndarray
            matrix holding feature vectors for all wikipedia articles
            obtained by factorising the tf-idf document-term matrix
            using SVD
        vectorizer: TfidfVectorizer
            sklearn.feature_extraction.text.TfidfVectorizer object used
            to calculate TFIDF scores
        svd_transformer: TruncatedSVD
            sklearn.decomposition.TruncatedSVD object used to factorise the
            tf-idf document-term matrix
    Methods:
        create_tf_idf_vectors(self, dataframe_filename):
            Create TFIDF vectors via sklearn's TfidfVectorizer on the
            dataset stored in the file give as dataframe_filename.

            dataframe_filename - The name of the file the dataset is stored in.

        rank_docs(self, query, docs, evaluate_component):
            Rank select documents to query with with cosine similarity of tf-idf values.

            Arguments:
                query - processed query that must be lemmatized, tokenized and have had
                    stop words removed or a named tuple of type Question
                docs - list of indices in dataset
                evaluate_component (default: False) - boolean value determining if
                    the evaluation setup is executed
            Returns:
                ranked list of wikipedia article identifiers corresponding to the indices
                given by docs
            Raises:
                TypeError if query is tuple and evaluate_component is True

        rank(self, query_tuple, query, evaluate_component):
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

    def __init__(self):
        self.index2key = {}
        self.vectorizer = None
        self.svd_transformer = None
        self.stop_words = spacy.load("en_core_web_sm").Defaults.stop_words
        self.doc_vecs = None
        self.truncated_doc_vecs = None

    def create_tf_idf_vectors(self, dataframe_filename: str):
        """
        Create TFIDF vectors by fitting model to dataset.

        The dataset is loaded from dataframe_filename and the vectorizer
        is trained on the dataset. After the training is complete the
        method fills index2vector for faster access.

        Arguments:
            dataframe_filename - The name of the file the dataset is stored in.
        """
        content_matrix = []
        content_matrix, self.index2key = load_corpus(dataframe_filename)
        self.index2key = {
            key: links.split(" ") for key, links in self.index2key.items()
        }
        # stop words are already removed
        self.vectorizer = TfidfVectorizer(min_df=5)  # default for now, fine-tune later
        self.vectorizer = self.vectorizer.fit(content_matrix)
        self.doc_vecs = self.vectorizer.transform(content_matrix)

        # Create truncated document matrix
        self.svd_transformer = TruncatedSVD(n_components=512)
        self.svd_transformer.fit(self.doc_vecs)
        self.truncated_doc_vecs = self.svd_transformer.transform(self.doc_vecs)

    def rank_docs(
        self,
        query: Union[namedtuple, List[str]],
        docs: List[int],
        evaluate_component: bool = False,
        max_docs: int = 10,
    ) -> List[str]:
        """
        Rank select documents to query with with cosine similarity of tf-idf values.

        Arguments:
            query - processed query that must be lemmatized, tokenized and have had
                stop words removed or a named tuple of type Question
            docs - list of indices in dataset
            evaluate_component (default: False) - boolean value determining if
                the evaluation setup is executed
            max_docs (default: 10) - maximum number of (merged) articles to return
        Returns:
            ranked list of wikipedia article identifiers corresponding to the indices
            given by docs
        Raises:
            TypeError if query is tuple and evaluate_component is True
        """
        if evaluate_component and not isinstance(query, list):
            raise TypeError(
                "namedTuple Question is mutuably exlusive with evaluation_component"
                "flag and processed query string!"
            )
        if evaluate_component:
            query = [" ".join(query)]
        else:
            query = [" ".join(query.terms)]
        query_vector = self.vectorizer.transform(query)
        cosine_similarities = linear_kernel(query_vector, self.doc_vecs[docs])
        cosine_similarities = cosine_similarities.flatten()
        related_docs_indices = cosine_similarities.argsort()[::-1][:max_docs]

        # return [link for index in related_docs_indices for link in self.index2key[str(index)]]
        return [" ".join(self.index2key[str(index)]) for index in related_docs_indices]

    def rank(
        self,
        query_tuple: Union[namedtuple, None] = None,
        query: List[str] = None,
        evaluate_component: bool = False,
        approximation: bool = True,
    ) -> List[str]:
        """Rank all documents to query with cosine similarity of tf-idf values

        Arguments:
            query_tuple (default: None) - namedTuple of Question as defined by
                question_parsing component, cannot be combined with query or
                evaluate_component flag
            query (default: None) - processed query that must be lemmatized,
                 tokenized and have had stop words removed, must be set with
                 evaluate_component
            evaluate_component (default: False) - boolean value determining if
                the evaluation setup is executed
            approximation (default: False) - boolean value determining if
                to use the truncated feature vectors instead of the full
                tf-idf vectors

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
        if not query_tuple:
            query = [" ".join(query)]
        else:
            query = [" ".join(query_tuple.terms)]

        # Adapted from https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity
        query_vector = self.vectorizer.transform(query)
        if approximation:
            # First, we calculate the approximate similarities using
            # from the truncated document-term matrix (this is fast)
            truncated_query_vector = self.svd_transformer.transform(query_vector)
            truncated_cosine_similarities = linear_kernel(
                truncated_query_vector, self.truncated_doc_vecs
            ).flatten()
            truncated_related_indices = truncated_cosine_similarities.argsort()[::-1]

            # We extract the 1000 highest-scoring documents and calculate
            # similarities using the full tf-idf document-term matrix
            best_truncated_indices = truncated_related_indices[:1000]
            cosine_similarities = linear_kernel(
                query_vector, self.doc_vecs[best_truncated_indices]
            )
            cosine_similarities = cosine_similarities.flatten()
            related_docs_indices = cosine_similarities.argsort()[::-1]

            # Finally we resubstitute the ordering of the 1000 highest-scoring
            # documents using truncated features by the ordering obtained by
            # using full tf-idf vectors.
            # Ordering for all other documents is unchanged
            truncated_related_indices[:1000] = best_truncated_indices[
                related_docs_indices
            ]
            related_docs_indices = truncated_related_indices
        else:
            # If we don't want to use truncated features (slow), we compute
            # all similarities using the full tif-idf document-term matrix
            cosine_similarities = linear_kernel(query_vector, self.doc_vecs).flatten()
            related_docs_indices = cosine_similarities.argsort()[::-1]

        return [self.index2key[str(index)] for index in related_docs_indices]
