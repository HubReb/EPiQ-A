# TODO:
#  1. train BM25 on train_article to extract
#  2. TEST BM25 + QA model on dev_article

import pickle
from os.path import isfile
from data_utils import load_csv
from gensim.summarization.bm25 import BM25

class Okapi_BM_25:
    def __init__(self, filename="trained_bm25.pkl"):

        if isfile(filename):
            self.bm25_model = self.load_model(filename)  # ?
        else:
            self.bm25_model = None

    def load_model(self, filename):
        """Load a trained bm25 model from a pickle file."""
        with open(filename, "rb") as f:
            return pickle.load(f)

    def fit(self, csv_path):
        """Fit gensim's BM25 model to data."""
        # articles: str

        dataframe = load_csv(csv_path)
        # print(dataframe.columns.values)
        # print(dataframe.Text_Proc.values)
        processed_articles = ''
        # loop throu numpy array
        for articles in dataframe.Text_Proc.values:
            processed_articles = articles + '\n' # a big string

        passages = [passage.strip() for passage in processed_articles.splitlines()]  # a list of sentences of the articles

        self.bm25_model = BM25(passages)

def train(datapath):
    """Train okapi BM25 model and save it"""
    BM25 = Okapi_BM_25()
    BM25.fit(datapath)
    # sav
    with open("trained_bm25.pkl", "wb") as f:
        pickle.dump(BM25, f)

if __name__ == "__main__":

    train('corpus_processed.csv')
