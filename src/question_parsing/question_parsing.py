# -*- coding: utf-8 -*-

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

#print("Initialising global variables for question parsing. This may take some time")
QuestionFields = [
    "terms",             # Relevant tokens of the question
    "original_terms",    # The original terms (including stopwords)
                         # no preprocessing
    #"synonyms",          # Synonyms to relevant tokens (for term augmentation)
                         ## -> combat sparsity!
    #"pos_tags",          # POS tags of relevant tokens
    #"named_entities",    # Named Entities in question
    #"focus"              # Main keyword in question, has large influence on
                         # the expected answer
]
Question = namedtuple("Question", QuestionFields)  # Datastructure to store
# questions
SpacyQuestion = spacy.tokens.doc.Doc
Token = spacy.tokens.token.Token
Tokens = List[Union[Token, str]]

SpacyWizard = spacy.load("en_core_web_sm")
# EmbeddingModel = gensim.load("glove-wiki-gigaword-300")
# EmbeddingModel = gensim.load("glove-wiki-gigaword-50")
Stemmer = nltk.stem.snowball.EnglishStemmer()
WordNetLemmatiser = nltk.stem.WordNetLemmatizer()
Stopwords = set(stopwords.words("english"))
# Set of question words found by a manual inspection of the NaturalQuestions
# Dataset and by consulting
# https://dictionary.cambridge.org/grammar/british-grammar/question-words
#QuestionWords = {
    #"what",
    #"what's",
    #"whats",
    #"what’s",
    #"when",
    #"when's",
    #"whens",
    #"when’s",
    #"where",
    #"where's",
    #"whereabouts",
    #"whether",
    #"whi",
    #"which",
    #"while",
    #"who",
    #"who's",
    #"whom",
    #"whos",
    #"whose",
    #"who’s",
    #"why",
    #"how",
#}


##############################################################################
# Helper functions                                                           #
##############################################################################


def make_spacy_representation(question: str) -> SpacyQuestion:
    return SpacyWizard(question)


# Adapted from
# https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
# https://spacy.io/api/annotation#pos-tagging
#def convert_spacy_pos_to_nltk(pos: str) -> str:
    #if pos == "ADJ":
        #return wordnet.ADJ
    #elif pos == "VERB":
        #return wordnet.VERB
    #elif pos == "NOUN":
        #return wordnet.NOUN
    #elif pos == "ADV":
        #return wordnet.ADV
    #else:
        #return ""


def postprocess_string(
    term: str,
    lemmatise: bool = False,
    stem: bool = False,
    tolower: bool = False,
    filter_stopwords: bool = False,
) -> str:
    """
    Performs preprocessing operations on the token level.
    Assumes str as input.
    This includes:
      * lemmatising
      * stemming
      * lowercasing
      * stopword filtering
    """
    assert not lemmatise or not stem, "Can't both stem and lemmatise"
    if tolower:
        term = term.lower()

    if filter_stopwords and term in Stopwords:
        return None

    if stem:
        return Stemmer.stem(term)  # Stemmer is a global variable

    elif lemmatise:
        # Find possible lemma (we don't know the pos tag, so we try all)
        for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
            lemma = WordNetLemmatiser.lemmatize(term, pos)
            # If we have found a POS-Tag that allows truncation, we assume
            # that it's the correct one
            if lemma != term:
                return lemma
        else:
            return term

    return term


def postprocess_spacy_token(
    term: Token,
    lemmatise: bool = False,
    stem: bool = False,
    tolower: bool = False,
    filter_stopwords: bool = False,
) -> str:
    """
    Performs preprocessing operations on the token level.
    Assumes spacy token as input.
    This includes:
      * lemmatising
      * stemming
      * lowercasing
      * stopword filtering
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
    """
    Performs normalisation operations on a sequence of tokens.
    This includes:
      * lemmatising
      * stemming
      * lowercasing
      * stopword filtering
    """
    processed_terms = []
    for term in terms:
        if isinstance(term, str):
            postprocessed_term = postprocess_string(term, **kwargs)

        elif isinstance(term, Token):
            postprocessed_term = postprocess_spacy_token(term, **kwargs)

        else:
            raise TypeError("Unknown token datatype {}".format(type(term)))

        if postprocessed_term is not None:
            processed_terms.append(postprocessed_term)

    return processed_terms


#def get_lemmas_from_wordnet_synset(synset):
    #"""
    #Extracts all lemmata from the given wordnet synset.
    #"""
    #all_lemmata = []
    #for lemma in synset.lemma_names():
        #if not lemma.isalnum():  # Filter compounds / genre names / ...
            #continue

        #if lemma in Stopwords:  # Filter stopwords
            #continue

        #all_lemmata.append(lemma)
    #return all_lemmata


#def get_hypernym_closure(synset):
    #"""
    #Extracts all (transitive) hypernyms of a given synset as str.
    #"""
    #all_hypernyms = []
    #for hypernym in synset.closure(lambda s: s.hypernyms()):
        #all_hypernyms.extend(get_lemmas_from_wordnet_synset(hypernym))
    #return all_hypernyms


#def get_hyponym_closure(synset):
    #"""
    #Extracts all (transitive) hyponyms of a given synset as str.
    #"""
    #all_hyponyms = []
    #for hyponym in synset.closure(lambda s: s.hyponyms()):
        #all_hyponyms.extend(get_lemmas_from_wordnet_synset(hyponym))
    #return all_hyponyms


#def find_root(question: SpacyQuestion) -> Token:
    #"""
    #Finds the root element of the question's dependency parse.
    #"""
    #for token in question:
        #if token.dep_ == "ROOT":
            #return token


#def get_noun_chunk(token: Token, question: SpacyQuestion) -> List[str]:
    #accepted_pos = ["NUM", "NOUN", "PROPN", "ADJ"]
    #for noun_chunk in question.noun_chunks:
        #if token in noun_chunk:
            #return [str(tken) for tken in noun_chunk
                    #if tken.pos_ in accepted_pos]

    #return [str(token)]


##############################################################################
# Extractor functions                                                        #
##############################################################################


#def get_wordnet_synonyms(term: str, pos: str, max_synonyms: int,
                         #include_hypernyms: bool = False,
                         #include_hyponyms: bool = False) -> List[str]:
    #"""
    #Returns all Lemmas from all WordNet synsets of `term`.
    #"""
    #all_synsets = wordnet.synsets(term, pos=pos)
    #all_synonyms = []
    
    #for synset in all_synsets:
        #all_synonyms.extend(get_lemmas_from_wordnet_synset(synset))
        
        #if include_hypernyms:
            #all_synonyms.extend(get_hypernym_closure(synset))
        
        #if include_hyponyms:
            #all_synonyms.extend(get_hyponym_closure(synset))
    
    #return all_synonyms[:min(max_synonyms, len(all_synonyms))]


#def get_word2vec_synonyms(term: str, pos: str, max_synonyms: int) -> List[str]:
    #"""
    #Retrieves the `max_synonyms` tokens most similar to `terms` from the
    #global Embedding model.
    #"""
    #try:
        #synonyms = EmbeddingModel.most_similar(term.lower(), topn=max_synonyms)
        #return [synonym for synonym, _ in synonyms if synonym not in Stopwords]
    #except KeyError:
        #return []


#def get_synonyms(
    #question: SpacyQuestion, method: str = None, max_synonyms: int = 10,
    #include_hypernyms: bool = False, include_hyponyms: bool = False
#) -> List[str]:
    #"""
    #Depending on `method`, invokes the correct synonym retrieval
    #function for all tokens in `question`.
    #"""
    #if method == "word2vec":
        #synonym_getter = get_word2vec_synonyms
    #elif method == "wordnet":
        #synonym_getter = partial(get_wordnet_synonyms,
                                 #include_hypernyms = include_hypernyms,
                                 #include_hyponyms = include_hyponyms)

    #all_synonyms = []
    #for token in question:
        #pos = convert_spacy_pos_to_nltk(token.pos_)
        #if str(token) not in Stopwords:  # Ignore stopwords
            #synonyms = synonym_getter(str(token), pos, max_synonyms)
            #synonyms = [synonym.lower() for synonym in synonyms]
            #all_synonyms.extend(synonyms)
    
    #question_tokens = set([str(token) for token in question
                           #if token not in Stopwords and \
                           #token in EmbeddingModel])
    #all_synonyms = list(set([synonym for synonym in all_synonyms
                             #if synonym not in question_tokens and \
                             #synonym not in Stopwords]))
    
    #if len(all_synonyms) == 0:
        #return all_synonyms
    
    ## Sort by embeddings space similarity
    #random.shuffle(all_synonyms)
    #if max_synonyms < len(all_synonyms) and question_tokens:
        #synonyms_in_embedding_model = [synonym for synonym in all_synonyms
                                       #if synonym in EmbeddingModel]
        #synonyms_notin_embedding_model = [synonym for synonym in all_synonyms
                                          #if synonym not in EmbeddingModel]
        #distances = [EmbeddingModel.n_similarity(question_tokens, [synonym])
                     #for synonym in synonyms_in_embedding_model]
        #_, sorted_synonyms = \
            #zip(*sorted(zip(distances, synonyms_in_embedding_model)))
        #all_synonyms = list(sorted_synonyms) + synonyms_notin_embedding_model

    #return list(set(all_synonyms))[:min(max_synonyms, len(all_synonyms))]


#def get_named_entities(question: SpacyQuestion) -> List[str]:
    #return [str(ent).lower() for ent in question.ents]


#def get_pos_tags(question: SpacyQuestion) -> List[str]:
    #return [str(token.pos_) for token in question]


#def get_focus(question: SpacyQuestion) -> str:
    #root = find_root(question)
    #noun_chunk = partial(get_noun_chunk, question=question)

    #nsubjs = []
    #dobjs = []

    ## 1st rule: Try to find nsubj or dobj
    #for token in root.children:
        ## Exclude question words:
        #if str(token).lower() in QuestionWords:
            #continue

        #elif token.pos_ == "PRON":
            #continue

        #if token.dep_ == "nsubj":
            #nsubjs.append(token)

        #elif token.dep_ == "dobj":
            #dobjs.append(token)

    #if nsubjs:
        #return noun_chunk(nsubjs[0])  # First subject

    #elif dobjs:
        #return noun_chunk(dobjs[0])  # First dobj

    ## Fallback: Choose first noun that is not a question word
    #for token in root.children:
        #for token in root.subtree:
            #if token.pos_ == "NOUN" and str(token) not in QuestionWords:
                #return noun_chunk(token)

    ## Fallback 2: Return root (although in this case we should really
    ##             question the question's sanity
    #return noun_chunk(root)


##############################################################################
# Main functions (to be called from outside)                                 #
##############################################################################


def parse_question(
    question: str,
    #max_synonyms: int = 10,
    #synonym_method: str = "wordnet",
    #include_hyponyms: bool = False,
    #include_hypernyms: bool = False,
    lemmatise: bool = True,
    stem: bool = False,
    tolower: bool = True,
    filter_stopwords: bool = True
) -> Question:
    """
    Main method for parsing a question.
    """
    spacy_question = make_spacy_representation(question)
    postprocessor = partial(
        postprocess,
        lemmatise=lemmatise,
        stem=stem,
        tolower=tolower,
        filter_stopwords=filter_stopwords,
    )

    return Question(
        terms=postprocessor(spacy_question),
        original_terms = [str(token).lower() for token in spacy_question],
        #pos_tags=get_pos_tags(spacy_question),
        #synonyms=postprocessor(
            #get_synonyms(spacy_question, synonym_method, max_synonyms,
                         #include_hyponyms, include_hypernyms)
        #),
        #named_entities=postprocessor(get_named_entities(spacy_question)),
        #focus=get_focus(spacy_question),
    )


if __name__ == "__main__":
    with open("test_questions.txt") as tf:
        for line in tf:
            question = line.strip()
            parse = parse_question(question) # include_hyponyms=True, include_hypernyms=True)

            print(question)
            # print(parse.synonyms)
            print()
