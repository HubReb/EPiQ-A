# TODO
#  add a only from bert predicted answer
import pickle
from os.path import isfile
from data_utils import load_csv, csv_to_list_of_passages
import spacy

class Okapi_BM_25():
    def __init__(self, csv_path, bm25_model_filename="trained_bm25.pkl"):

        # we have a trained model
        if isfile(bm25_model_filename):
            print('Loading trained bm25 model.')
            model = self.load_model(bm25_model_filename)
            self.passages = model.passages
            self.nlp = model.nlp
            self.bm25_model = model.bm25_model

        # we need to start from scratch
        else:
            self.passages = csv_to_list_of_passages(csv_path)  # a list of passages of the articles
            # print(self.passages)
            # print("passage: ", len(self.passages))
            self.nlp = spacy.load('en_core_web_sm')
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def tokenize_question(self, question, lemmatize=False):  #### we dont need on testing?
        """
        :param sentence:
        :return:
             - if lemmatize == True: a list of tokenized and lemmatized strings
             - else: a list of tokenized strings
        """

        if lemmatize:
            return [token.lemma_.lower() for token in self.nlp(question)]
        return [str(token).lower() for token in self.nlp(question)]

    def get_n_top_passages(self, n_passages, question: str):
        """ONLY WORK IF THE RETRIEVED DOCUMENTS USE NEWLINES TO SEPARATE PARAGRAPHS.
        :param articles: the joint top n answer relevant articles(str)
        :param question: preoprocessed question(str)
        :return: jointed top n passages from the articles as a final article(str)
        """

        bm25_scores = self.bm25_model.get_scores(self.tokenize_question(question))

        # get top n relevant passages
        best_docs = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i])[n_passages * (-1):]

        questionContext = ""
        for fr in best_docs:
            # print(self.passages[fr], "\n\n")
            questionContext = questionContext + "".join(self.passages[fr])

        return questionContext

    def load_model(self, filename):
        """Load a trained bm25 model from a pickle file."""
        with open(filename, "rb") as f:
            return pickle.load(f)