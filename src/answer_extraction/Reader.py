#!/usr/bin/env python3
# TODO: run data on
import pickle
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from data_utils import load_csv
from passage_BM25 import Okapi_BM_25
import spacy
import time


class AnswerExtracter():
    """Extract exact answer from the extracted passages
    Using the pretrained models from the transformers library for:
        - tokenization: to tokenize the question and the question context
        - question answering:  to find the tokens for the answer
    """

    def __init__(self, model="distilbert"):
        self.nlp = spacy.load('en_core_web_sm')
        if model == "bert":
            self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_token_type_ids=True)
            self.model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')


    def getAnswer(self, question: str, questionContext: str):
        ###we’ll need to make all the vectors the same size by padding shorter sentences with the token id 0

        doc = self.nlp(question)
        questionContext = ' '.join([str(token) for token in doc][:511]).strip()

        inputs = self.tokenizer(question, questionContext, return_tensors='pt')

        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])
        outputs = self.model(**inputs, start_positions=start_positions, end_positions=end_positions)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Answer span predictor
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        encoding = self.tokenizer.encode_plus(question, questionContext)
        inputs = encoding['input_ids']  # Token embeddings
        tokens = self.tokenizer.convert_ids_to_tokens(inputs)  # input tokens

        # convert answer tokens back to string and return the result
        answer = ' '.join(tokens[start_index:end_index + 1])
        print('question: ', question)
        print('answer: ', answer)
        return answer

    def getAnswer_original(self, question: str, questionContext: str):

        encodings = self.tokenizer.encode_plus(question, questionContext)

        inputIds, attentionMask = encodings["input_ids"], encodings["attention_mask"]

        scoresStart, scoresEnd = self.model(torch.tensor([inputIds]), attention_mask=torch.tensor([attentionMask]))

        tokens = inputIds[torch.argmax(scoresStart): torch.argmax(scoresEnd) + 1]
        answerTokens = self.tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)

        answer = self.tokenizer.convert_tokens_to_string(answerTokens)

        return answer


if __name__ == "__main__":
    start = time.time()

    # top n result of BM25
    n_passages = 3

    # Local
    # articles_csv_path = "./data/processed_article_corpus.csv"
    # bm25 = Okapi_BM_25(articles_csv_path, bm25_model_filename="trained_bm25_2.pkl")
    # Last
    articles_csv_path = "./processed_merged_wiki_text.csv"
    bm25 = Okapi_BM_25(articles_csv_path, bm25_model_filename="trained_bm25_2.pkl")

    answerExtracter = AnswerExtracter(model="distilbert")

    # Local
    # question_dev_dataframe = load_csv("./data/nq_dev_short.csv")
    # Last
    question_dev_dataframe = load_csv("/proj/epiqa/EPiQ-A/data/natural_questions_train.csv")

    question_dev_dataframe["Question Context"] = ''
    question_dev_dataframe["Predicted Answer"] = ''

    print("Predicting answers...")
    for i, row in question_dev_dataframe.iterrows():
        questionContext = bm25.get_n_top_passages(n_passages, row["Question"])
        question_dev_dataframe.at[i, "Question Context"] = str(questionContext)
        question = row["Question"]

        predictedAnswer = answerExtracter.getAnswer(question, questionContext)
        # print(type(predictedAnswer), predictedAnswer)
        question_dev_dataframe.at[i, "Predicted Answer"] = predictedAnswer
    # Local
    # question_dev_dataframe.to_csv(f'./data/predicted_answers.csv', encoding='utf-8', index=False)
    # Last
    question_dev_dataframe.to_csv(f'./predicted_answers.csv', encoding='utf-8', index=False)

    print(f'\nData frame written to predicted_answers.csv')
    end = time.time()
    print("\nRum time: ", end - start)
