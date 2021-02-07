# TODO:
#  1. train BM25 on train_article to extract
#  2. TEST BM25 + QA model on dev_article
import time
import pickle
from gensim.summarization.bm25 import BM25
from passage_BM25 import Okapi_BM_25

def train(datapath):
    """Train okapi BM25 model and save it"""
    start = time.time()
    OBM25 = Okapi_BM_25(datapath, "")
    OBM25.bm25_model = BM25(OBM25.passages)
    with open("trained_bm25.pkl", "wb") as f:
        pickle.dump(OBM25, f)
    end = time.time()
    print("Training completed. Model has been saved as trained_bm25.pkl \nRum time: ", end - start)

if __name__ == "__main__":

    # Local
    # train('./data/processed_article_corpus.csv')

    # Last
    train('./processed_merged_wiki_text.csv')