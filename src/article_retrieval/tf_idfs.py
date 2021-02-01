#!/usr/bin/env python3


""" Define model to rank documents relative to a query with cosine similarity of TFIDF vectors"""

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.utils.extmath import safe_sparse_dot
import numpy as np


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
        index2vector: dictionary
            mapping from the index of a document in the dataset to the
            corresponding TFIDF vector.
        model- spacy model, currently en_core_web_sm
        stop_words: set
            set of the stop words defined in the model
        self.vectorizer:
            sklearn.feature_extraction.text.TfidfVectorizer object used
            to calculate TFIDF scores
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

    def __init__(self, model=spacy.load("en_core_web_sm")):
        """
        Parameters:
            model: spacy model to use for lemmatization and stop word
                removal
        """
        self.index2key = {}
        self.index2vector = []
        self.model = model
        self.stop_words = self.model.Defaults.stop_words
        self.vectorizer = None

    def create_tf_idf_vectors(self, dataframe_filename):
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
        # stop words are already removed
        self.vectorizer = TfidfVectorizer(min_df=5)  # default for now, fine-tune later
        self.vectorizer = self.vectorizer.fit(content_matrix)
        doc_vecs = self.vectorizer.transform(content_matrix)
        for vector in doc_vecs:
            self.index2vector.append(vector)

    def rank_docs(self, docs, query, evaluate_component=False):
        """
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
        """
        if evaluate_component and not isinstance(query, list):
            raise TypeError(
                "namedTuple Question is mutuably exlusive with evaluation_component"
                "flag and processed query string!"
            )
        similarities = []
        if evaluate_component:
            query = [" ".join(query)]
        else:
            query = " ".join(query.terms)
        query_vector = self.vectorizer.transform(query)
        for doc in docs:
            document_vector = self.index2vector[doc]
            similarities.append((self.index2key[str(doc)], safe_sparse_dot(query_vector, document_vector.T)[0][0]))
        ranked_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
        ranked_docs = [doc_sim[0] for doc_sim in ranked_sims]
        return ranked_docs

    def rank(self, query_tuple=None, query=None, evaluate_component=False):
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
        similarities = []
        if not query_tuple:
            query = [" ".join(query)]
        else:
            query = query_tuple.terms
        query_vector = self.vectorizer.transform(query)
        for index, doc in enumerate(self.index2vector):
            # handle sparse vectors correctly
            similarities.append((self.index2key[str(index)], safe_sparse_dot(query_vector, doc.T, dense_output=True)[0][0]))
        ranked_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
        ranked_docs = [doc_sim[0] for doc_sim in ranked_sims]
        return ranked_docs
