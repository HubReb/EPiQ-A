import argparse

from time import time
from evaluation import evaluate
from pipeline import CombinedModel
from article_retrieval.config import BM25_MODEL, TFIDF_MODEL


parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    action="store",
    default="evaluate",
    type=str,
    choices=["evaluate", "interactive"],
    help="Determines the program you want to run",
)
parser.add_argument(
    "--article-retrieval-model",
    action="store",
    default="tfidf",
    type=str,
    choices=["bm", "tfidf"],
    help="Determines the model you want to use for article retrieval",
)
parser.add_argument(
    "--article-retrieval-model-filename",
    action="store",
    default=TFIDF_MODEL,
    type=str,
    choices=[BM25_MODEL, TFIDF_MODEL],
    help="Determines file you want to load your articel " "retrieval model from",
)
parser.add_argument(
    "--no-index",
    action="store_false",
    help="Switches inverted index off",
)
parser.add_argument(
    "--no-retrieval",
    action="store_false",
    help="Switches article retrieval off, retrieval based " "only on paragraphs",
)
parser.add_argument(
    "--num-articles",
    action="store",
    default=10,
    type=int,
    help="Number of articles to retrieve",
)
parser.add_argument(
    "--paragraph-level",
    action="store",
    default="paragraph",
    type=str,
    choices=["paragraph", "sentence"],
    help="Determines the paragraph splitting level",
)
parser.add_argument(
    "--num-paragraphs",
    action="store",
    default=10,
    type=int,
    help="Number of paragraphs to retrieve",
)
parser.add_argument(
    "--max-context-size",
    action="store",
    default=400,
    type=int,
    help="Maximum tokens per context (for answer extraction)",
)
parser.add_argument(
    "--transformers-model-name",
    action="store",
    default=None,
    help="Determines the transformers model to use for answer extraction",
)
parser.add_argument(
    "--include_synonyms",
    action="store_true",
    default=False,
    help="include synonyms of question words in query",
)

if __name__ == "__main__":
    args = parser.parse_args()
    print("Loading model. We're here to serve your wishes in just 2 or 3 hours.")
    model = CombinedModel(
        article_retrieval_model=args.article_retrieval_model,
        model_filename=args.article_retrieval_model_filename,
        article_retrieval_use_index=args.no_index,
        retrieve_articles=args.no_retrieval,
        num_articles=args.num_articles,
        answer_extraction_model_name=args.transformers_model_name,
        paragraph_level=args.paragraph_level,
        n_top_paragraphs=args.num_paragraphs,
        max_context_size=args.max_context_size,
        include_synonyms=args.include_synonyms,
    )
    if args.mode == "interactive":
        print("This is the interactive Q/A shell. How can I serve you today?")
        print()
        print("If you want to leave me, type in `exit`\n")
        while True:
            question = input("What's your question?:> ")
            if question == "exit":
                print("Shutdown. In Windows speed")
                break
            starttime = time()
            answer = model.get_answer(question)
            stoptime = time()
            print("Answer:", answer)
            print("This only took us {:.2f} seconds".format(stoptime - starttime))
            print()

    elif args.mode == "evaluate":
        evaluate(model)
