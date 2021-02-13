import pandas as pd

from question_parsing.question_parsing import parse_question
from article_retrieval.evaluate_baseline import load_utilities_for_bm, load_utilities_for_tfidf
from article_retrieval import gensim_bm25
from article_retrieval.gensim_bm25 import Okapi25
from article_retrieval.query_index import query_index
from article_retrieval.article_index import ArticlesFromTitleMentions
from article_retrieval.config import BM25_MODEL, TFIDF_MODEL

# from answer_extraction.Reader import AnswerExtracter


class CombinedModel:
    def __init__(self, article_retrieval_model: str, dataset: str, model_filename: str):
        self.question_parser = parse_question
        self.dataset = pd.read_csv(dataset)

        if article_retrieval_model == "bm":
            bm_model, bm_inverted_index = load_utilities_for_bm(
                model_filename,
                "article_retrieval/inverted_index.json"
            )
            self.article_model = bm_model
            self.inverted_index = bm_inverted_index
        elif article_retrieval_model == "tfidf":
            tfidf_model, tfidf_inverted_index = load_utilities_for_tfidf(
                    model_filename,
                    "article_retrieval/inverted_index.json"
            )
            self.article_model = tfidf_model
            self.inverted_index = tfidf_inverted_index
        else:
            raise ValueError("Unknown model: {}".format(article_retrieval_model))
        
        self.get_articles_from_titles = \
            ArticlesFromTitleMentions("article_retrieval/article_title_index.json")
        self.get_articles_from_titles = \
            self.get_articles_from_titles.get_articles_with_title_mentions

        # self.answer_extraction_model = AnswerExtracter()


    def get_answer(self, query: str):
        parsed_query = self.question_parser(question=query)
        docs, _ = query_index(parsed_query, self.inverted_index, self.article_model)
        ranked_ids = self.article_model.rank_docs(parsed_query, docs)
        ranked_ids.extend(self.get_articles_from_titles(parsed_query))
        ranked_ids = list(set(ranked_ids))
        
        print(parsed_query)
        print(self.get_articles_from_titles(parsed_query))

        # get text from ids and keep ranking
        top_paragraphs = []
        for ranked_id in ranked_ids:
            top_paragraphs.append(str(self.dataset[(self.dataset.Wikipedia_ID == ranked_id)]["Text"].values))
        print(len(top_paragraphs))

        #possible_ansers = []
        #for context in top_paragraphs:
            #answer = self.answer_extraction_model.getAnswer(query, context)
            #possible_ansers.append(answer)

        # Irgendein Code, um eine gute Answer zurückzugeben
        # raise NotImplementedError


if __name__ == "__main__":
    model = CombinedModel("tfidf", "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv", TFIDF_MODEL)
    model.get_answer("Where is Eiffel tower")
    print()
    model = CombinedModel("bm", "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv", BM25_MODEL)
    model.get_answer("Where is Eiffel tower")
