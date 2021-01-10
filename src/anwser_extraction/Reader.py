from gensim.summarization.bm25 import BM25
import torch
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import spacy
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

class ContextRetriever:

    """Searching for relevant context(Context retriever)
    Preprocess: Tokenize, Lemmatize(optional)
    Using BM25 to rank a list of passages based on a given query.
    Extract the top N results from BM25 and build a paragraph out of all those N sentences.

    :param nlp: spacy
    :param n_passage: select the n top relevant passages by BM25
    :param preprocess: default  = True.
            - True: tokenize and lemmatize the sentences(for real case)
            - False: only tokenize the sentences(for testing)
    """

    def __init__(self, nlp, n_passages, preprocess = True):
        self.nlp = nlp
        self.n_passages = n_passages
        self.preprocess = preprocess

    def tokenize(self, sentence): #### we dont need on testing!
        """
        :param sentence: a sentence string
        :return:
             - if preprocess == True: a list of tokenized and lemmatized strings
             - else: a list of tokenized strings
        """

        if self.preprocess == True:
            return [token.lemma_.lower() for token in self.nlp(sentence)]
        else:
            return [token.lower() for token in self.nlp(sentence)]

    def get_n_top_passages(self, articles: str, question: str):
        """
        :param articles: the joint top n answer relevant articles(str)
        :param question: preoprocessed question(str)
        :return: jointed top n passages from the articles as a final article(str)
        """

        # doc = self.nlp(articles)

        passages = [passage.strip() for passage in articles.splitlines()] # a list of sentences of the articles

        # a list of lists of tokens(a list of sentences)
        documents = [self.tokenize(passage) for passage in passages] # self.tokenize(sent): a list of tokens

        bm25_scores = BM25(documents).get_scores(self.tokenize(question))

        # get top n relevant passages
        best_docs = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i])[self.n_passages*(-1):]

        questionContext = ""
        for fr in best_docs:
            questionContext = questionContext + " ".join(documents[fr])

        return questionContext


class AnswerExtracter:
    """Extract exact answer from the extracted passages
    Using the pretrained models from the transformers library for:
        - tokenization: to tokenize the question and the question context
        - question answering:  to find the tokens for the answer
    """

    def getAnswer(self, question, questionContext, model="DistilBERT"):
        ###we’ll need to make all the vectors the same size by padding shorter sentences with the token id 0

        if model == "BERT":
            model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_token_type_ids=True)
            model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

        inputs = tokenizer(question, questionContext, return_tensors='pt')
        start_positions = torch.tensor([1])
        end_positions = torch.tensor([3])

        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # answer span predictor
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        encoding = tokenizer.encode_plus(question, questionContext)
        inputs = encoding['input_ids']  # Token embeddings
        tokens = tokenizer.convert_ids_to_tokens(inputs)  # input tokens

        # convert answer tokens back to string and return the result
        answer = ' '.join(tokens[start_index:end_index + 1])
        return answer

if __name__ == '__main__':

    articles = "The predominant language is Cantonese, a variety of Chinese originating in Guangdong. It is spoken by 94.6 per cent of the population, 88.9 per cent as a first language and 5.7 per cent as a second language. Slightly over half the population (53.2 per cent) speaks English, the other official language; 4.3 per cent are native speakers, and 48.9 per cent speak English as a second language Code-switching, mixing English and Cantonese in informal conversation, is common among the bilingual population. \nPost-handover governments have promoted Mandarin, which is currently about as prevalent as English; 48.6 per cent of the population speaks Mandarin, with 1.9 per cent native speakers and 46.7 per cent speaking it as a second language. \nTraditional Chinese characters are used in writing, rather than the simplified characters used on the mainland. "
    question = "what's the language of hongkong"

    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    contextRetriever = ContextRetriever(nlp, 1, preprocess = True)  # top n result of BM25

    questionContext = contextRetriever.get_n_top_passages(articles, question)

    answerExtracter = AnswerExtracter()
    answer = answerExtracter.getAnswer(question, questionContext) # model = ''
    #print(questionContext)

    print ("Question: ", question)
    print('Text: ', articles)
    print("Question Context: ", questionContext)

    print ("Anwser: ", answer)


