import pandas as pd

from question_parsing.question_parsing import parse_question
from article_retrieval.evaluate_baseline import load_utilities_for_bm, load_utilities_for_tfidf
from article_retrieval import gensim_bm25
from article_retrieval.gensim_bm25 import Okapi25
from article_retrieval.query_index import query_index

from anwser_extraction.Reader import AnswerExtracter


class CombinedModel:
    def __init__(self, article_retrieval_model: str, dataset: str):
        self.question_parser = parse_question
        self.dataset = pd.read_csv(dataset)

        if article_retrieval_model == "bm":
            bm_model, bm_inverted_index = load_utilities_for_bm(
                "article_retrieval/okapibm25.pkl",
                "/home/rebekka/Studium/master/ITA/project/EPiQ-A/src/article_retrieval/inverted_index.json"
            )
            self.article_model = bm_model
            self.inverted_index = bm_inverted_index
        elif article_retrieval_model == "tfidf":
            tfidf_model, tfidf_inverted_index = load_utilities_for_tfidf(
                    "article_retrieval/tfidfmodel.pkl",
                    "/home/rebekka/Studium/master/ITA/project/EPiQ-A/src/article_retrieval/inverted_index.json"
            )
            self.article_model = tfidf_model
            self.inverted_index = tfidf_inverted_index
        else:
            raise ValueError("Unknown model: {}".format(article_retrieval_model))

        self.answer_extraction_model = AnswerExtracter()


    def get_answer(self, query: str):
        parsed_query = self.question_parser(question=query)
        docs, _ = query_index(parsed_query, self.inverted_index, self.article_model)
        ranked_ids = self.article_model.rank_docs(parsed_query, docs)[:10]

        # get text from ids and keep ranking
        top_paragraphs = []
        for ranked_id in ranked_ids:
            top_paragraphs.append(str(self.dataset[(self.dataset.Wikipedia_ID == ranked_id)]["Text"].values))

        possible_ansers = []
        for context in top_paragraphs:
            answer = self.answer_extraction_model.getAnswer(query, context)
            possible_ansers.append(answer)

        # Irgendein Code, um eine gute Answer zurückzugeben
        raise NotImplementedError


if __name__ == "__main__":
    model = CombinedModel("bm", "../data/nq_train_wiki_text.csv")
    model.get_answer("Who is George W. Bush?")
