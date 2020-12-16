#!/usr/bin/env python3


""" Define model to rank documents relative to a query with cosine similarity of TFIDF vectors"""

import pickle

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from data_utils import load_corpus


class TFIDFmodel:
    """
    A document ranker that utilizes cosine similarity between TF-IDF vectors.

    The class uses sklearn.feature_extraction.text TfidfVectorizer to create
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

        rank_docs(self, query, docs):
            Rank select documents to query with with cosine similarity of tf-idf values.

            docs - list of indices in dataset
            query - processed query that must be lemmatized, tokenized and have had
                stop words removed

            Returns:
                 ranked list of wikipedia article identifiers

        rank(self, query):
            Rank all documents to query with cosine similarity of tf-idf values

            query - processed query that must be lemmatized, tokenized and have had
                stop words removed

            Returns:
                ranked list of wikipedia article identifiers
    """

    def __init__(self, model=spacy.load("en_core_web_sm")):
        """
        Parameters:
            model: spacy model to use for lemmatization and stop word
                removal
        """
        self.index2key = {}
        self.index2vector = {}
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
        self.vectorizer = TfidfVectorizer(min_df=5)
        self.vectorizer = self.vectorizer.fit(content_matrix)
        doc_vecs = self.vectorizer.transform(content_matrix)
        for index, vector in enumerate(doc_vecs):
            self.index2vector[self.index2key[index]] = vector

    def rank_docs(self, docs, query):
        """
        Rank select documents to query with with cosine similarity of tf-idf values.

        Arguments:
            docs - list of indices in dataset
            query - processed query that must be lemmatized, tokenized and have had
                stop words removed

        Returns:
            ranked list of wikipedia article identifiers corresponding to the indices
            given by docs
        """
        similarities = []
        query_vector = self.vectorizer.transform(query)
        for doc in docs:
            document_vector = self.index2vector[doc]
            similarities.append((doc, cosine_similarity(query_vector, document_vector)))
        ranked_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
        ranked_docs = [doc_sim[0] for doc_sim in ranked_sims]
        return [self.index2key[doc] for doc in ranked_docs]

    def rank(self, query):
        """Rank all documents to query with cosine similarity of tf-idf values

        Arguments:
            query - processed query that must be lemmatized, tokenized and have had
                stop words removed

        Returns:
            ranked list of wikipedia article identifiers corresponding to the indices
            of all documents in the dataset
        """

        similarities = []
        query_vector = self.vectorizer.transform(query)
        for index, doc in self.index2vector.items():
            similarities.append((index, cosine_similarity(query_vector, doc)))
        ranked_sims = sorted(similarities, key=lambda x: x[1], reverse=True)
        ranked_docs = [doc_sim[0] for doc_sim in ranked_sims]
        return [self.index2key[doc] for doc in ranked_docs]
