import spacy
import nltk
import random

import gensim.downloader as gensim

from collections import namedtuple
from functools import partial
from typing import List, Union
from nltk.corpus import wordnet
from nltk.corpus import stopwords


##############################################################################
# Global variables                                                           #
##############################################################################

QuestionFields = [
    'terms',
    'synonyms',
    'named_entities',
    'question_type',
    'focus',
    'focus_modifiers'
    ]
Question = namedtuple('Question', QuestionFields)
SpacyQuestion = spacy.tokens.doc.Doc
Token = spacy.tokens.token.Token
Tokens = List[Union[Token, str]]

SpacyWizard = spacy.load("en_core_web_sm")  # TODO: Change to larger model
EmbeddingModel = gensim.load("glove-twitter-25")  #TODO: Change to larger
Stemmer = nltk.stem.snowball.EnglishStemmer()
WordNetLemmatiser = nltk.stem.WordNetLemmatizer()
Stopwords = set(stopwords.words('english'))


##############################################################################
# Helper functions                                                           #
##############################################################################

def make_spacy_representation(question: str) -> SpacyQuestion:
    return SpacyWizard(question)


# Adapted from
# https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
# https://spacy.io/api/annotation#pos-tagging
def convert_spacy_pos_to_nltk(pos: str) -> str:
    if pos == 'ADJ':
        return wordnet.ADJ
    elif pos == "VERB":
        return wordnet.VERB
    elif pos == "NOUN":
        return wordnet.NOUN
    elif pos == "ADV":
        return wordnet.ADV
    else:
        return ''
    

def postprocess_string(
        term: str, lemmatise=False, stem=False, tolower=False,
        filter_stopwords=False) -> str:
    """
    Docstring
    """
    assert not lemmatise or not stem, "Can't both stem and lemmatise"
    if tolower:
        term = term.lower()
    
    if filter_stopwords and term in Stopwords:
        return None
    
    if stem:
        return Stemmer.stem(term)
    
    elif lemmatise:
        for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
            lemma = WordNetLemmatiser.lemmatize(term, pos)
            if lemma != term:
                return lemma
        else:
            return term
    
    return term


def postprocess_spacy_token(
        term, lemmatise=False, stem=False, tolower=False,
        filter_stopwords=False) -> str:
    """
    Docstring
    """
    assert not lemmatise or not stem, "Can't both stem and lemmatise"
    term_as_str = str(term)
    if filter_stopwords and term_as_str in Stopwords:
        return None
    
    if stem:
        term_as_str = Stemmer.stem(term_as_str)
    
    elif lemmatise:
        term_as_str = term.lemma_
    
    if tolower:
        term_as_str = term_as_str.lower()
    
    return term_as_str


def postprocess(terms: Tokens, **kwargs) -> List[str]:
    processed_terms = []
    for term in terms:
        if isinstance(term, str):
            postprocessed_term = postprocess_string(term, **kwargs)
        
        elif isinstance(term, Token):
            postprocessed_term = postprocess_spacy_token(term, **kwargs)
        
        if postprocessed_term is not None:
            processed_terms.append(postprocessed_term)
    
    return processed_terms


##############################################################################
# Extractor functions                                                        #
##############################################################################

def get_wordnet_synonyms(term: str, pos: str, max_synonyms: int) -> List[str]:
    all_synsets = wordnet.synsets(term, pos=pos)
    all_synonyms = []
    # TODO: Hyponyms/Hypernyms?
    for synset in all_synsets:
        for lemma in synset.lemma_names():
            if lemma.isalnum() and lemma not in Stopwords:
                all_synonyms.append(lemma)
    
    # TODO: Smarter selection
    random.shuffle(all_synonyms)
    return all_synonyms[:min(max_synonyms, len(all_synonyms))]


def get_word2vec_synonyms(term: str, pos: str, max_synonyms: int) -> List[str]:
    synonyms = EmbeddingModel.most_similar(term.lower(), topn=max_synonyms)
    return [synonym for synonym, _ in synonyms if synonym not in Stopwords]


def get_synonyms(question: SpacyQuestion, method: str = None, max_synonyms: int = 10) -> List[str]:
    if method == "word2vec":
        synonym_getter = get_word2vec_synonyms
    else:
        synonym_getter = get_wordnet_synonyms
    
    all_synonyms = []
    for token in question:
        pos = convert_spacy_pos_to_nltk(token.pos_)
        synonyms = synonym_getter(str(token), pos, max_synonyms)
        all_synonyms.extend(synonyms)
    
    return list(set(all_synonyms))


def get_named_entities(question: SpacyQuestion) -> List[str]:
    return [str(ent).lower() for ent in question.ents]


def get_question_type() -> str:
    return None


def get_focus() -> str:
    return None


##############################################################################
# Main functions (to be called from outside)                                 #
##############################################################################


def parse_question(
        question: str, max_synonyms=10, synonym_method='word2vec',
        lemmatise=False, stem=False, tolower=False,
        filter_stopwords=False) -> Question:
    """
    Docstring
    """
    spacy_question = make_spacy_representation(question)
    postprocessor = partial(
        postprocess,
        lemmatise=lemmatise,
        stem=stem, tolower=tolower,
        filter_stopwords=filter_stopwords
        )
    
    return Question(
        terms = postprocessor(spacy_question),
        synonyms = postprocessor(
            get_synonyms(spacy_question, synonym_method, max_synonyms)
            ),
        named_entities = postprocessor(get_named_entities(spacy_question)),
        question_type = get_question_type(),
        focus = get_focus(),
        focus_modifiers = None
        )


if __name__ == '__main__':
    qtext = "Who was America's most successful president?"
    parse = parse_question(qtext, stem=True)
    
    print(parse)
