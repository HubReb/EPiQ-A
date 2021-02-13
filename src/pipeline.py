import time
import pandas as pd

from question_parsing.question_parsing import parse_question
from article_retrieval.evaluate_baseline import load_utilities_for_bm, load_utilities_for_tfidf
from article_retrieval import gensim_bm25
from article_retrieval.gensim_bm25 import Okapi25
from article_retrieval.query_index import query_index
from article_retrieval.article_index import ArticlesFromTitleMentions
from article_retrieval.config import BM25_MODEL, TFIDF_MODEL

from answer_extraction.reader_working import AnswerFromContext
from answer_extraction.data_utils import load_csv


class CombinedModel:
    def __init__(self, article_retrieval_model: str, dataset: str, model_filename: str,
                 article_retrieval_use_index: bool = True,
                 retrieve_articles: bool = True):
        self.article_retrieval_use_index = article_retrieval_use_index
        self.retrieve_articles = retrieve_articles
        self.max_articles = 10
        
        self.question_parser = parse_question
        
        print("Loading articles")
        self.dataset = pd.read_csv(dataset)
        
        print("Loading article retrieval model")
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
        
        print("Loading article title index model")
        self.get_articles_from_titles = \
            ArticlesFromTitleMentions("article_retrieval/article_title_index.json")
        self.get_articles_from_titles = \
            self.get_articles_from_titles.get_articles_with_title_mentions
        
        print("Loading (pretrained) answer extraction model")
        self.answer_extraction_model = AnswerFromContext(load_paragraphs=not self.retrieve_articles)


    def get_answer(self, query: str):
        parsed_query = self.question_parser(question=query)
        if self.retrieve_articles:
            if self.article_retrieval_use_index:
                docs, _ = query_index(parsed_query, self.inverted_index, self.article_model)
                ranked_ids = self.article_model.rank_docs(parsed_query, docs)
            else:
                ranked_ids_ = self.article_model.rank(parsed_query)
                ranked_ids_ = ranked_ids_[:self.max_articles]
                ranked_ids = []
                for links in ranked_ids_:
                    ranked_ids.extend(links)
        
            ranked_ids.extend(self.get_articles_from_titles(parsed_query))
            ranked_ids = list(set(ranked_ids))

            # get text from ids and keep ranking
            top_articles = []
            for ranked_id in ranked_ids:
                top_articles.append(str(self.dataset[(self.dataset.Wikipedia_ID == ranked_id)]["Text"].values))
        
            answer = self.answer_extraction_model.get_answer(parsed_query, top_articles)
        
        else:
            answer = self.answer_extraction_model.get_answer(parsed_query)
        
        return answer


if __name__ == "__main__":
    model = CombinedModel("tfidf", "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv",
                          TFIDF_MODEL, article_retrieval_use_index=False, retrieve_articles=False)
    # model.get_answer("who had most wins in nfl")
    # print()
    # model = CombinedModel("bm", "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv", BM25_MODEL)
    # model.get_answer("who had most wins in nfl")
    
    question_dev_dataframe = load_csv("../data/natural_questions_train.csv")

    print("Predicting answers...")
    for i, row in question_dev_dataframe.iterrows():
        starttime = time.time()
        question = row["Question"]
        answer = model.get_answer(question)
        endtime = time.time()
        
        print("Question:", question)
        print("Answer:", answer)
        print("Needed:", endtime-starttime, "seconds")
        print()
