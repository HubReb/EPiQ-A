import os
import csv
import sys
import time
import pickle
import numpy as np

from tqdm import tqdm
from data_utils import load_csv
from transformers import pipeline
from gensim.summarization.bm25 import BM25


if __name__ == "__main__":
    start = time.time()

    # top n result of BM25
    n_passages = 3

    # Local
    # articles_csv_path = "./data/processed_article_corpus.csv"
    # bm25 = Okapi_BM_25(articles_csv_path, bm25_model_filename="trained_bm25_2.pkl")
    # Last
    print("Loading paragraphs")
    articles_csv_path = "../../data/article_retrieval/nq_dev_train_wiki_text_merged.csv"
    paragraphs = []
    csv.field_size_limit(sys.maxsize)
    with open(articles_csv_path) as af:
        for _, _, text in tqdm(csv.reader(af, delimiter=',')):
            for paragraph in text.split("\n\n"):
                paragraphs.append(paragraph.split())
    
    print("Loading BM25 model")
    if os.path.isfile("bm25_paragraph_model.pkl"):
        with open("bm25_paragraph_model.pkl", 'rb') as bm25_file:
            bm25 = pickle.load(bm25_file)
    else:
        print("Training new model. This may take some time.")
        bm25 = BM25(paragraphs)
        with open("bm25_paragraph_model.pkl", 'wb') as bm25_file:
            pickle.dump(bm25, bm25_file)
        
    print(bm25.average_idf)
    print("Loading pretrained Q/A model")
    answerExtracter = pipeline("question-answering")

    # Local
    # question_dev_dataframe = load_csv("./data/nq_dev_short.csv")
    # Last
    question_dev_dataframe = load_csv("../../data/natural_questions_train.csv")

    question_dev_dataframe["Question Context"] = ''
    question_dev_dataframe["Predicted Answer"] = ''

    print("Predicting answers...")
    for i, row in question_dev_dataframe.iterrows():
        question = row["Question"]
        scores = bm25.get_scores(question.split())
        top_scoring_paragraph_indices = np.argsort(scores)[::-1][:n_passages]
        top_scoring_paragraphs = [paragraphs[index][:150] for index in top_scoring_paragraph_indices]
        top_scoring_paragraphs = [" ".join(paragraph) for paragraph in top_scoring_paragraphs]
        
        context = "\n\n".join(top_scoring_paragraphs)
        predictedAnswer = answerExtracter(question=question, context=context)
    
        print(question)
        print(predictedAnswer)
        print()

    end = time.time()
    print("\nRum time: ", end - start)
