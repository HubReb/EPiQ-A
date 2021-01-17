import pandas as pd

from question_parsing.question_parsing import parse_question
from article_retrieval.evaluate_baseline import load_utilities_for_bm, load_utilities_for_tfidf
from answer_extraction.Reader import AnswerExtracter


class CombinedModel:
    def __init__(self, article_retrieval_model: str, dataset: str):
        self.question_parser = parse_question
        self.dataset = pd.read_csv(dataset)

        if article_retrieval_model == "bm":
            bm_model, bm_inverted_index = load_utilities_for_bm()
            self.article_model = bm_model
            self.inverted_index = bm_inverted_index
        elif self.article_retrieval_model == "tfidf":
            tfidf_model, tfidf_inverted_index = load_utilities_for_tfidf()
            self.article_model = tfidf_model
            self.inverted_index = tfidf_inverted_index
        else:
            raise ValueError("Unknown model: {}".format(article_retrieval_model))

        self.answer_extraction_model = AnswerExtracter()


    def get_answer(self, query: str):
        parsed_query = self.question_parser(question=query)
        ranked_ids = self.article_model.rank(parsed_query)

        # get text from ids
        top_paragraphs = self.dataset["WikipediaID"].loc[ranked_ids]

        possible_ansers = []
        for context in top_paragraphs:
            answer = self.answer_extraction_model.getAnswer(query, context)
            possible_ansers.append(answer)

        # Irgendein Code, um eine gute Answer zurückzugeben
        raise NotImplementedError