import os
import csv
import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from answer_extraction.data_utils import load_csv
from transformers import pipeline
from gensim.summarization.bm25 import BM25

ARTICLESPATH = "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv"
B25MODELFILE = "answer_extraction_bm25_paragraph_model.pkl"


class AnswerFromContext:
    def __init__(self, load_paragraphs):
        self.model = pipeline("question-answering")
        self.n_top_paragraphs = 10
        self.max_context_size = 400
        
        self.load_paragraphs = load_paragraphs
        if load_paragraphs:
            print("Working without retrieved articles -- load data")
            self.bm25, self.paragraphs = self.load_data()
    
    def load_data(self):
        print("Answer extraction model - Loading paragraphs")
        articles_csv_path = ARTICLESPATH
        paragraphs = []
        csv.field_size_limit(sys.maxsize)
        
        with open(articles_csv_path) as af:
            for _, _, text in tqdm(csv.reader(af, delimiter=',')):
                for paragraph in text.split("\n\n"):
                    paragraphs.append(paragraph.split())
    
        print("Answer extraction model - Loading BM25 model")
        if os.path.isfile(B25MODELFILE):
            with open(B25MODELFILE, 'rb') as bm25_file:
                bm25 = pickle.load(bm25_file)
        else:
            print("Training new model. This may take some time.")
            bm25 = BM25(paragraphs)
            with open(B25MODELFILE, 'wb') as bm25_file:
                pickle.dump(bm25, bm25_file)
        
        return bm25, paragraphs


    def get_best_paragraphs_from_articles(self, question, articles):
        # Prepare paragraphs
        paragraphs = []
        for article in articles:
            for paragraph in article.split("\n\n"):
                paragraphs.append([token.strip().lower() for token in paragraph.split()])
        
        # Train temporary BM25 model on paragrpahs
        bm25 = BM25(paragraphs)

        # Extract top-scoring paragraphs
        scores = bm25.get_scores(question.original_terms)
        top_scoring_paragraph_indices = np.argsort(scores)[::-1][:self.n_top_paragraphs]
        top_scoring_paragraphs = [paragraphs[index] for index in top_scoring_paragraph_indices]

        return top_scoring_paragraphs
    
    def get_best_paragraphs_from_all_data(self, question):
        scores = self.bm25.get_scores(question.original_terms)
        top_scoring_paragraph_indices = np.argsort(scores)[::-1][:self.n_top_paragraphs]
        top_scoring_paragraphs = [self.paragraphs[index] for index in top_scoring_paragraph_indices]
        
        return top_scoring_paragraphs


    def get_answer(self, question, articles=None):
        print("Retrieving paragraphs")
        if self.load_paragraphs:
            top_scoring_paragraphs = self.get_best_paragraphs_from_all_data(question)
        else:
            top_scoring_paragraphs = self.get_best_paragraphs_from_articles(question, articles)
        
        question = " ".join(question.original_terms)

        answers = []
        for i, paragraph in enumerate(top_scoring_paragraphs):
            print("Processing paragraph {}/{}".format(i + 1, len(top_scoring_paragraphs)), end='\r')
            window_boundaries = list(range(0, len(paragraph), self.max_context_size)) + [len(paragraph)]
            for start, stop in zip(window_boundaries[:-1], window_boundaries[1:]):
                window = paragraph[max(0, start - 20): stop]
                context = " ".join(window)
                answer = self.model(question=question, context=context)
                answers.append(answer)

        return next(iter(sorted(answers, key=lambda a: a['score'], reverse=True)))['answer']


if __name__ == "__main__":
    start = time.time()

    # top n result of BM25
    n_passages = 3

    print("Loading paragraphs")
    # Local
    # articles_csv_path = "./data/processed_article_corpus.csv"

    # Last
    # - Not preprocessed
    articles_csv_path = "../../build/data/article_retrieval/nq_dev_train_wiki_text_merged.csv"
    # - Preprocessed
    # articles_csv_path = "../../build/data/answer_extraction/processed_merged_wiki_text.csv"
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
    # question_dataframe = load_csv("./data/nq_dev_short.csv")
    # Last
    question_dataframe = load_csv("../../build/data/natural_questions_train.csv")

    question_dataframe["Context"] = ''
    question_dataframe["Predicted Answer"] = ''

    print("Predicting answers...")
    for i, row in question_dataframe.iterrows():
        question = row["Question"]
        scores = bm25.get_scores(question.split())
        top_scoring_paragraph_indices = np.argsort(scores)[::-1][:n_passages]
        top_scoring_paragraphs = [paragraphs[index][:150] for index in top_scoring_paragraph_indices]
        top_scoring_paragraphs = [" ".join(paragraph) for paragraph in top_scoring_paragraphs]

        context = "\n\n".join(top_scoring_paragraphs)
        predictedAnswer = answerExtracter(question=question, context=context)

        question_dataframe.at[i, "Context"] = context
        question_dataframe.at[i, "Predicted Answer"] = predictedAnswer['answer']

        print(question)
        print(predictedAnswer)
        print()

    # Local
    # question_dataframe.to_csv(f'./predicted_answers.csv', encoding='utf-8', index=False)
    # Last
    question_dataframe.to_csv('../../build/data/predicted_answers.csv', encoding='utf-8', index=False)
    print('Predicted Answer saved to ../../build/data/predicted_answers.csv')
    end = time.time()
    print("\nRum time: ", end - start)
