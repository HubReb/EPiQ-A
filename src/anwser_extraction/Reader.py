# We built our component upon the blog-post
# https://programmerbackpack.com/bert-nlp-using-distilbert-to-build-a-question-answering-system/
# and adapt the freely available code to our case

from gensim.summarization.bm25 import BM25
import torch
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import spacy

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer



class QuestionProcessor:
    """
    Remove stop words, Tokenize, Part of Speech tags,
    Keep only the essential parts: nouns, proper nouns, and adjectives

    Example:
        Original question: “Whats the capital city of France?”
     => Processed question: “capital city France”
    """

    def __init__(self, nlp):
        self.pos = ["NOUN", "PROPN", "ADJ"]
        self.nlp = nlp


    def process(self, text):
        tokens = self.nlp(text)
        return ' '.join(token.text for token in tokens if token.pos_ in self.pos)


class ContextRetriever:

    """
    Searching for relevant context(Context retriever)

    Preprocess: Tokenize, Lemmatize

    Using BM25 to rank a list of documents based on a given query.

    Extract the top N results from BM25 and build a paragraph out of all those N sentences.
    """

    def __init__(self, nlp, numberOfResults):
        self.nlp = nlp
        self.numberOfResults = numberOfResults

    def tokenize(self, sentence):
        return [token.lemma_ for token in self.nlp(sentence)]


    def getContext(self, sentences, question):
        """
        :param sentences:
        :param question:
        :return:
        """

        documents = []
        for sent in sentences:
            documents.append(self.tokenize(sent))

        bm25 = BM25(documents)

        scores = bm25.get_scores(self.tokenize(question))
        results = {}
        for index, score in enumerate(scores):
            results[index] = score

        sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
        results_list = list(sorted_results.keys())
        final_results = results_list if len(results_list) < self.numberOfResults else results_list[:self.numberOfResults]
        questionContext = ""
        for final_result in final_results:
            questionContext = questionContext + " ".join(documents[final_result])
        return questionContext


class AnswerRetriever:
    """
    Using the pretrained models from the transformers library for:
        - tokenization: to tokenize the question and the question context
        - question answering:  to find the tokens for the answer

    Convert answer tokens back to string and return the result
    """

    def getAnswer_1(self, question, questionContext):


        # Model
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        encoding = tokenizer.encode_plus(text=question, text_pair=questionContext, add_special=True)

        inputs = encoding['input_ids']  # Token embeddings
        sentence_embedding = encoding['token_type_ids']  # Segment embeddings
        tokens = tokenizer.convert_ids_to_tokens(inputs)  # input tokens
        start_scores, end_scores = model(input_ids=torch.tensor([inputs]),
                                         token_type_ids=torch.tensor([sentence_embedding]))
        start_index = torch.argmax(start_scores)

        end_index = torch.argmax(end_scores)

        answer = ' '.join(tokens[start_index:end_index + 1])

        corrected_answer = ''

        for word in answer.split():

            # If it's a subword token
            if word[0:2] == '##':
                corrected_answer += word[2:]
            else:
                corrected_answer += ' ' + word

        return corrected_answer



    def getAnswer_2(self, question, questionContext):
        # Alternative 1

        # executing these commands for the first time initiates a download of the
        # model weights to ~/.cache/torch/transformers/
        tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
        model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")
        # 1. TOKENIZE THE INPUT
        inputs = tokenizer.encode_plus(question, questionContext, return_tensors="pt")

        # 2. OBTAIN MODEL SCORES
        # a span predictor
        answer_start_scores, answer_end_scores = model(**inputs)
        print(answer_start_scores) # => start_logits
        answer_start = torch.argmax(answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score
        # 3. GET THE ANSWER SPAN
        # grab all the tokens in the span and convert tokens back to words!
        anwser = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
        return anwser



    def getAnswer(self, question, questionContext):

        # worked but weird anwser span
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

        input_dict = tokenizer(question, questionContext, return_tensors='tf')
        #print(input_dict)

        outputs = model(input_dict)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0]: tf.math.argmax(end_logits, 1)[0] + 1])

        return answer

    def getAnswer_3(self, question, questionContext):
        #ORIGINAL
        distilBertTokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_token_type_ids=True)
        distilBertForQuestionAnswering = DistilBertForQuestionAnswering.from_pretrained(
            'distilbert-base-uncased-distilled-squad')

        encodings = distilBertTokenizer.encode_plus(question, questionContext)

        inputIds, attentionMask = encodings["input_ids"], encodings["attention_mask"]


        scoresStart, scoresEnd = distilBertForQuestionAnswering(torch.tensor([inputIds]),
                                                                attention_mask=torch.tensor([attentionMask]))

        tokens = inputIds[torch.argmax(scoresStart): torch.argmax(scoresEnd) + 1]

        answerTokens = distilBertTokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)
        return distilBertTokenizer.convert_tokens_to_string(answerTokens)

if __name__ == '__main__':


    text = "Paris is the capital and most populous city of France, with an estimated population of 2,148,271 residents as of 2020, in an area of 105 square kilometres . Since the 17th century, Paris has been one of Europe's major centres of finance, diplomacy, commerce, fashion, science and arts. The City of Paris is the centre and seat of government of the Île-de-France, or Paris Region, which has an estimated official 2020 population of 12,278,210, or about 18 percent of the population of France. "
    originalQuestion = "What is the capital city of France?"

    nlp = spacy.load('en_core_web_sm')

    nlp.add_pipe(nlp.create_pipe('sentenci'
                                 'zer'))
    doc = nlp(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    #print(sentences)
    questionProcessor = QuestionProcessor(nlp)
    contextRetriever = ContextRetriever(nlp, 10)

    questionContext = contextRetriever.getContext(sentences, questionProcessor.process(originalQuestion))

    answerRetriever = AnswerRetriever()
    answer = answerRetriever.getAnswer(originalQuestion, questionContext)
    #print(questionContext)

    print ("Original Question: ", originalQuestion)
    print ("Preprocessed Question: ", questionProcessor.process(originalQuestion))

    print ("Anwser: ", answer)


