## Question Processing
The question processing component extracts all information needed for further processing from the question. A question is given as a string.
The extracted information includes:

  * Question keywords: Either all tokens or only non-stopword tokens
  * POS-Tags of the retained tokens
  * Synonyms or related words of all non-stopword tokens
  * Named Entities
  * Question focus: One or a few tokens that represent the main theme or actor of the question
  * Question category using the taxonomy from [1] (not yet implemented)


### Tokenisation, POS tagging, NER
Tokenisation, POS tagging, and NER are performed using the spacy library [2] ( https://spacy.io/ ).

### Synonyms
Currently, two ways of extractiong synonyms are supported:

  * Retrieving all lemmata from WordNet [3] synsets related to a token
  * Retrieving a fixed number of closest tokens in a vector-space model

For retrieving similar terms from a vector-space model, we use gensim [4] ( https://radimrehurek.com/gensim/ ). Gensim also provides pre-trained embedding models ( https://github.com/RaRe-Technologies/gensim-data ).

In upcoming work, we would like to combine both WordNet and the vector space model, for example by ranking WordNet synonyms by similarity in the embedding space. Also, we plan to include Hyponyms and Hypernyms from WordNet.

### Question focus
The question focus is a span from the question (usually corresponding to 1 single noun or a multiword expression like compounds). The question focus indicates at the main theme or actor of the question. The question focus therefore hints at the expected answer. Previous work [5, 6, inter alios] has shown that determining the question focus improves question interpretation.

For example, according to [5], the question focus in "What mystery writer penned '...the glory that was Greece, and the grandeur thatwas Rome'?" is "mystery writer", because this is the main characteristic of the expected answer.

In previous work, the question focus is extracted from a constituent parse of the question using a set of rules, most notably the so-called "Collins rules" [7, 8]. As a baseline, we have implemented a different set of rules that operate on the constituency parse of the question. A constituency parse is also provided by gensim.

First, we determine the root of the constituency parse. Then, the following rules are applied:

  * If the root has a nominal subject as direct child which is not a question word, the nominal subject is returned.
  * If the the root has a direct object as child which is not a question word, the direct object is returned.
  * Otherwise, the leftmost noun in the question is returned.
  * If the question does not contain nouns, the root is returned.

Instead of returning only one token, we extract the whole noun phrase of the question focus. Of all tokens in the noun phrase, we only keep nouns, proper nouns, adjectives, and numbers. This helps with resolving parsing problems arising from multiword expressions and named entities.

In upcoming work, we plan to improve handling of named entities, multiword expressions, and disfluencies. Optimally, we can manipulate the question string so that the quality of the constituent parses improve.

### Question category
Disclaimer: The functionality for predicting the question category is not yet implemented.

We ´classify each question using the fine-grained taxonomy from [1] (obtained from https://cogcomp.seas.upenn.edu/Data/QA/QC/ ). This taxonomy indicates the category of the expected answer. For example, a question asking for a definition is given the label "definition". Other possible labels are "city", "vehicle", "size", amongst others.

We plan to train a classifier on the training set from https://cogcomp.seas.upenn.edu/Data/QA/QC/. Then, we plan to use the trained classifier for classifying new questions. Previous work [5] reports accuracies of approximately 90% on the test set for Linear Models using various features.

### Evaluation
For evaluating synonym extraction and question focus identification, we currently use 100 questions from the Natural Questions dataset to conduct manual evaluations. For evaluating the question category classification, we use the test set from https://cogcomp.seas.upenn.edu/Data/QA/QC/ .


### References
[1] Xin Li and Dan Roth. 2002. *Learning Question Classifiers*. 19th International Conference on Computational Linguistics, COLING 2002, Howard International House and Academia Sinica, Taipei, Taiwan, August 24 - September 1.

[2] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. *spaCy: Industrial-strength Natural Language Processing in Python*. Zenondo.

[3] Christiane Fellbaum. 1998. *A Semantic Network of English: The Mother of All WordNets*. In Computational Humanities 32, pages 209--220.

[4] Radim Rehurek and Petr Sojka. 2010. *Software Framework for Topic Modelling with Large Corpora*. In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks, pages 45--50.

[5] Harish Tayyar Madabushi and Mark Lee. 2016. *High Accuracy Rule-based Question Classification using Question Syntax and Semantics*. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 1220-1230 Osaka, Japan.

[6] Zhiheng Huang, Marcus Thint, Zengchang Qin. 2008. *Question classification using head words and their hypernyms*. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, EMNLP'08, pages 927-936, Stroudsburg, PA, USA. Association for Computational Linguistics.

[7] Michael Collins. 2003. *Head-Driven Statistical Models for Natural Language Parsing*. In Computational Linguistics 29, vol. 4, pages 589-637.

[8] Joao Silva, Luisa Coheur, Ana Cristina Mendes, and Andreas Wichert. 2011. *From symbolic to sub-symbolic information in question classification*. In Artificial Intellelligence Review, 35, pages 137-154.