import time
import sacrebleu
import editdistance

from pipeline import CombinedModel
from article_retrieval.config import BM25_MODEL, TFIDF_MODEL
from answer_extraction.data_utils import load_csv

def jaccard_index(predicted, correct):
    predicted, correct = set(predicted), set(correct)
    union = len(set.union(predicted, correct))
    intersection = len(set.intersection(predicted, correct))
    
    return intersection/union


def bleu(predicted, correct):
    return sacrebleu.sentence_bleu(predicted, [correct]).score


def word_error_rate(predicted, correct):
    error_rate = editdistance.eval(predicted, correct)
    error_rate /= len(correct)
    return error_rate

def exact_match(predicted, correct):
    return float(int(predicted == correct))


if __name__ == '__main__':
    print("Loading model")
    model = CombinedModel("tfidf", "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv",
                          TFIDF_MODEL, article_retrieval_use_index=True, retrieve_articles=True)
    # model.get_answer("who had most wins in nfl")
    # print()
    # model = CombinedModel("bm", "../data/article_retrieval/nq_dev_train_wiki_text_merged.csv", BM25_MODEL)
    # model.get_answer("who had most wins in nfl")
    
    print("Loading questions")
    question_dev_dataframe = load_csv("../data/natural_questions_train.csv")

    print("Predicting answers...")
    
    sum_jaccard_index = 0.0
    sum_bleu = 0.0
    sum_wer = 0.0
    sum_exact_match = 0.0
    sum_time = 0.0
    
    print("Starting evaluation")
    print()
    for i, row in question_dev_dataframe.iterrows():
        starttime = time.time()
        question = row["Question"]
        predicted_answer = model.get_answer(question)
        endtime = time.time()
        
        correct_answer = row["Answer"]
        
        sum_jaccard_index += jaccard_index(predicted_answer.split(), correct_answer.split())
        sum_bleu += bleu(predicted_answer, correct_answer)
        sum_wer += word_error_rate(predicted_answer.split(), correct_answer.split())
        sum_exact_match += exact_match(predicted_answer.split(), correct_answer.split())
        sum_time += (endtime - starttime)
        
        print(" "*180, end='\r')
        print("Avg. Jaccard-Index: {:.2f}\tAvg. BLEU: {:.2f}\tAvg. WER: {:.2f}\tAvg. Exact Match: {:.2f}\tAvg. Time/Question: {:.2f}s\tTotal questions: {}"\
            .format(sum_jaccard_index / (i+1), sum_bleu/(i+1), sum_wer/(i+1), sum_exact_match/(i+1), sum_time/(i+1), i+1), end='\r')
