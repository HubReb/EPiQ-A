# EPiQ-A

## Team members
* Marinco Möbius (moebius@cl.uni-heidelberg.de)
* Jin Huang (huang@cl.uni-heidelberg.de)
* Leander Girrbach (girrbach@cl.uni-heidelberg.de)
* Rebekka Hubert (hubert@cl.uni-heidelberg.de)

## Existing code
Each pipeline step has its own README that states what pre-existing code/data was used.

## Requirements

All requirements are given in requirements.txt

## Installation

First install the requirements, change into the `src` folder  and then run `pip install . --editable` to continue development. Otherwise run `pip install .`

## Usage Guide
### Data Preparation & Model Training
Please run `TODO.sh`. Data preparation may take up to ? hours.

### Running Experiments
 1. Navigate into the `src/` folder
 2. Run `python main.py --mode evaluate`. Other options are `--mode interactive` for aksing your own questions and `--mode evaluate_article_retrieval` for evaluating only the article retrieval module.
 3. Our implementation supports multiple modes of retrieving relevant articles and paragraphs, as well as configuring many important parameters. Run `python main.py --help` for an overview.

## Project State

### Planning State

At this stage, the datasets have been preprocessed. We have also implemented the basic components of our application, though all can be further improved.

In detail, this means that we can preprocess questions, extract named entities from questions and heuristically determine the question focus and synonyms to keywords in the question. Also we can build an inverted index from a collection of documents. By applying ranking mechanisms with regard to a question we can extract the documents from the collection most likely to contain the answer the question. Finally, we can extract the most relevant passages from the retrieved documents, extracting the best answer span from the relevant passages is still in progress.

The interaction between the pipeline parts is still fragile at best and requires further improvement. Due to this, we have not yet started on our overall goal: The error-awareness. Furthermore, we have not evaluated the pipeline in total, but have taken a look at the performance of the singular components.

### Future Planning

Our next goals are to connect the components of the pipeline to enable a seamless, easy end-to-end usage as  advised by our mentor and to extract the best answer span form the most relevant passages. We believe this can be easily done before the new year starts.
Then, we will address our main goal: The error-correction in the different components. We believe this will take the most of our remaining time.
Finally, we may improve our different pipeline components by either adding more features or implementing less simple algorithms to improve our  results.


### High-level Architecture Description:
There are two parts to the project: Data preprocessing and the application pipeline. The pipeline is constructed of three different parts: The construction of the semantic representation of the query, the retrieval of articles that may contain the answer to the query and the extraction of the answer from the retrieved articles. Whereas the parsing of each question and the extraction of the corresponding answer are written in single scripts, the retrieval of the articles requires its own submodule  due to the different, interchangeable components in this step.
In the following, we will describe each pipeline step in detail. Given a question, it is necessary to extract information needed for retrieving the answer. This includes primarily finding keywords, e.g. by tokenising, filtering stopwords, and optionally lemmatising.

Besides question keywords, recognising named entities is useful both for extracting answers and retrieving documents. Also, to deal with synonymy and sparsity (questions are generally short), we collect synonyms or closely related words to all keywords of the question.


Previous work on question answering has shown that finding out the question type, e.g. whether the question asks for a definition, number, etc. can help identifying good answers. To this end, the main focus keyword of a question is extracted. For example, the focus keyword in "What is the longest river in Africa" is "river", because it directly specifies the expected answer, i.e. name of a river.

Thus we describe this step now in more detail:

#### Question Processing
The question processing component extracts all information needed for further processing from the question. A question is given as a string.
The extracted information includes:

  * Question keywords: Either all tokens or only non-stopword tokens
  * POS-Tags of the retained tokens
  * Synonyms or related words of all non-stopword tokens
  * Named Entities
  * Question focus: One or a few tokens that represent the main theme or actor of the question
  * Question category using the taxonomy from [1] (not yet implemented)


##### Tokenisation, POS tagging, NER
Tokenisation, POS tagging, and NER are performed using the spacy library [2] ( https://spacy.io/ ). This information has proven useful and is thus included in the current version.

##### Synonyms - deprecated due to failing to improve results
Currently, two ways of extractiong synonyms are supported:

  * Retrieving all lemmata from WordNet [3] synsets related to a token
  * Retrieving a fixed number of closest tokens in a vector-space model

For retrieving similar terms from a vector-space model, we use gensim [4] ( https://radimrehurek.com/gensim/ ). Gensim also provides pre-trained embedding models ( https://github.com/RaRe-Technologies/gensim-data ).

In upcoming work, we would like to combine both WordNet and the vector space model, for example by ranking WordNet synonyms by similarity in the embedding space. Also, we plan to include Hyponyms and Hypernyms from WordNet.

##### Question focus - deprecated due to yielding mostly wrong focuses
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

##### Evaluation
For evaluating synonym extraction and question focus identification, we currently use 100 questions from the Natural Questions dataset to conduct manual evaluations. For evaluating the question category classification, we use the test set from https://cogcomp.seas.upenn.edu/Data/QA/QC/ .




The article retrieval component works independently from this first step as this additional information is not required to extract the relevant articles. 
Note that the natural questions dataset has new articles in the development set, which is our test set. We need to add these articles to our training set, otherwise we cannot retrieve the correct articles as they would be unknown.

We first build an inverted index from the articles in the entire dataset: We lemmatize each tokenized word in the dataset. Then we remove stop words. The remaining words are used to build the index. Each word maps to a list of documents that contain this word.

For querying the document collection, the words in the lemmatised query are mapped to the articles via the inverted index and the retrieved articles are ranked with respect to the query. Currently, we have two ranking, bag-of-words methods implemented: The TF-IDF Weighting introduced in the lecture with cosine similarity to the TF-IDF vector of the query for ranking and Okapi BM25. Note that our current implementation of TF-IDF uses [sklearns](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) default parameters.
For the Okapi BM25 we use the implementation provided by the python package [gensim](https://radimrehurek.com/gensim_3.8.3/summarization/bm25.html).
Regardless of the chosen ranking model we pass the top ten ranked articles to the answer extraction component.
All further detail is given in the submodules folder `article_retrieval`.


There are two main tasks to extract accurate answers to questions in the retrieved articles: 1) retrieve informative/relevant paragraphs in the article, and 2) extract the answer from those paragraphs. 

To extract the relative paragraphs from the articles, we use BM25 (implemented by gensim). It takes a query and sorts the paragraphs based on how relevant they are for the query. Then the top N paragraphs will be extracted by BM25 and we will build a article out of all those top N paragraphs as the final informative answer containing context for the question.

For question answering, transformers provides models that are fine-tuned checkpoints of DistilBERT(a simpler and faster version of Google's BERT model which still keeps most of the original model performance) or BERT, they are fine-tuned using knowledge distillation on SQuAD v1.1. They are models with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute span start logits and span end logits).

The pre-trained models distilbert-base-uncased-squad2 and bert-large-uncased-whole-word-masking-finetuned-squad from transformers can tokenize the question context(answer) and find the tokens for the answer. This model can answer "yes/no questions" with a sentence instead of "yes/no" answers, thus we might also solve this problem.
We built our component upon the blog-post https://programmerbackpack.com/bert-nlp-using-distilbert-to-build-a-question-answering-system/ and adapt the freely available code to our case.


### Experiments 

At the moment we test each pipeline independently in order to assure a level of quality of each component.

So far, we have evaluated the article retriever with the Okapi BM25 ranking model automatically. Because our dataset contains only one correct article, we focus on the rank the module assigns to the correct document: We calculate R-precision.
Because our goal is for the one correct article to be ranked as high as possible we also compute Mean Reciprocal Rank (MMR).

| Model       |       Metric |   Test |
|-------------|-------------:|-------:|
| Okapi BM25  |  R-precision | 0.0034 |
| TF-IDF cos. |  R-precision | 0.0034 |
| Okapi BM25  |          MMR | 0.0069 |
| TF-IDF cos. |          MMR | 0.3884 |
| Okapi BM25  |    Recall@10 | 0.0116 |
| TF-IDF cos. |    Recall@10 | 0.5781 |
| Okapi BM25  | Precision@10 | 0.0012 |
| TF-IDF cos. | Precision@10 | 0.0012 |

These results show that we may easily improve upon our current method. As soon as we fine-tune the model, our results will improve to some extent. Note that though the TF-IDF weighting ranking method achieves far better results, it is more than 10x slower than the Opaki BM25 weighting ranking model.

## Data Analysis

### Data Sources 

* [WikiMovies](https://research.fb.com/downloads/babi/):  
This dataset was dropped as it doesn't contain a direct association of question to Wikipedia article.

* [ConvQuestions](https://convex.mpi-inf.mpg.de/):  
We still intend to use this dataset, but it shrank substantially after preprocessing. We noticed that the dataset contains a lot of duplicates. 

* [NaturalQuestions](https://ai.google.com/research/NaturalQuestions/dataset):  
The main data source for our project. 

### Preprocessing 

#### Conv Questions

Like the name indicates, this dataset consists out of a number of **conversation threads** (**Question** & **Answer**), each belonging to one of the following domains: *books*, *movies*, *music*, *soccer*, *series*. Each conversation has multiple questions -- we only extract the first question as the subsequent questions are dependent on the first question and could only be interpreted in context.

Each conversation has a **seed entity** which started the conversation. For each entity they include the **link to Wikidata** (e.g. [American Hustle](https://www.wikidata.org/wiki/Q9013673) ). As we want an association of question/answer to Wikipedia, we first need to resolve this. Most Wikidata pages provide such a link to their respective Wikipedia page. We use the [query service](https://query.wikidata.org/) of Wikidata to receive the Wikipedia URL and then download and clean the article. 

From an Wikipedia article we parse the infobox, the headings and text and tables from the paragraphs. We remove HTML markup and footnote superscripts to receive plain text, additionally we skip unwanted sections (*References*, *Footnotes*, *Notes*, *See also*, ...). At the end we tokenize the article.

We organized the final data into a set of triples (question, answer, Wikipedia-URL) and a lookup of Wikipedia-URL and its text (tokens).

#### Natural Questions

This rather large dataset (~ 300k/8k/8k train/dev/test) provides for each **question** a **long answer** and a **Wikipedia-URL** from which the answer was selected, if possible they also included a **short answer**. 

They also include the HTML of the Wikipedia pages and the size of the training set amounts to 41 GB in total, but they also provide a simplified version with only the text (4GB in total). This simplified version is only available for the train set. It is therefore needed to convert the dev set into the same format. The test set is not publicly available. They also provide a conversion script, but it is geared for their pipeline and it would have been more work to integrate it than to do this ourselves.  

We only focus on questions with short answers or Yes/No answers (marked separately in the dataset). We discard the rest. We resolve the answer spans to text and clean the provided Wikipedia article in the same way as for the ConvQuestions dataset (remove HTML and unwanted sections).

The data is organized again in the same format as for the previous dataset.

### Basic Statistics

Number of documents in ConvQuestions before/after preprocessing (reduction due to duplicates):
|     CQ | train | dev  | test |
|-------:|------:|------|------|
| before |  6720 | 2240 | 2240 |
|  after |   414 | 136  | 140  |

Number of documents in NaturalQuestions before/after preprocessing (reduction due to filter of only short answers):
|     NQ |  train | dev  |
|-------:|-------:|------|
| before | 307373 | 7830 |
| after  | 110724 | 2662 |


85 \% of questions in ConvQuestions contain one of the following question words:
<img src="/figures/CQ_question_word_types.png" width="420" />

88 \% of questions in NaturalQuestions contain one of the following question words:
<img src="/figures/NQ_question_word_types.png" width="420" />

Top 10 of the most frequent auxiliary verbs or other words in NaturalQuestions when there is no w-question:
<img src="/figures/NQ_aux_types.png" width="420" />

Distribution of Named Entity in answer in dependence of question word (NaturalQuestions dataset):

* **who**
![Distribution of Named Entities for who-questions](figures/who.png "Distribution of Named Entities for who-questions")

* **when**
![Distribution of Named Entities for when-questions](figures/when.png "Distribution of Named Entities for when-questions")

* **where**
![Distribution of Named Entities for where-questions](figures/where.png "Distribution of Named Entities for where-questions")

* **what**
![Distribution of Named Entities for what-questions](figures/what.png "Distribution of Named Entities for what-questions")

* **which**
![Distribution of Named Entities for which-questions](figures/which.png "Distribution of Named Entities for which-questions")

### Examples

#### ConvQuestions

| Question                                                                      | Answer        | Wikipedia_ID                                      |
|:------------------------------------------------------------------------------|:--------------|:--------------------------------------------------|
| Which band produced the album known as the "dark side of the moon"?           | Pink Floyd    | https://en.wikipedia.org/wiki/Pink_Floyd          |
| Which actor starred in the tv series Two And A Half Men and Anger Management? | Charlie Sheen | https://en.wikipedia.org/wiki/Two_and_a_Half_Men  |
| Who wrote The Shining?                                                        | Stephen King  | https://en.wikipedia.org/wiki/The_Shining_(novel) |
| Taylor Swift's record label?                                                  | Big Machine Records                | https://en.wikipedia.org/wiki/Taylor_Swift          |

#### Natural Questions

| Question                                            | Answer                                                                                                          | Wikipedia_ID                                                                                                  |
|:----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|
| what do the 3 dots mean in math                     | the therefore sign ( ∴ ) is generally used before a logical consequence , such as the conclusion of a syllogism | https://en.wikipedia.org//w/index.php?title=Therefore_sign&amp;oldid=815234923                                |
| who is playing the halftime show at super bowl 2016 | Coldplay with special guest performers Beyoncé and Bruno Mars                                                   | https://en.wikipedia.org//w/index.php?title=Super_Bowl_50_halftime_show&amp;oldid=823813276                   |
| who won the 2017 sports personality of the year     | Mo Farah                                                                                                        | https://en.wikipedia.org//w/index.php?title=2017_BBC_Sports_Personality_of_the_Year_Award&amp;oldid=816169117 |
| name of black man in to kill a mockingbird                         | Thomas `` Tom '' Robinson                                                                                                                                | https://en.wikipedia.org//w/index.php?title=List_of_To_Kill_a_Mockingbird_characters&amp;oldid=835936451                               |


## References

[1] Xin Li and Dan Roth. 2002. *Learning Question Classifiers*. 19th International Conference on Computational Linguistics, COLING 2002, Howard International House and Academia Sinica, Taipei, Taiwan, August 24 - September 1.

[2] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. *spaCy: Industrial-strength Natural Language Processing in Python*. Zenondo.

[3] Christiane Fellbaum. 1998. *A Semantic Network of English: The Mother of All WordNets*. In Computational Humanities 32, pages 209--220.

[4] Radim Rehurek and Petr Sojka. 2010. *Software Framework for Topic Modelling with Large Corpora*. In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks, pages 45--50.

[5] Harish Tayyar Madabushi and Mark Lee. 2016. *High Accuracy Rule-based Question Classification using Question Syntax and Semantics*. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 1220-1230 Osaka, Japan.

[6] Zhiheng Huang, Marcus Thint, Zengchang Qin. 2008. *Question classification using head words and their hypernyms*. In Proceedings of the Conference on Empirical Methods in Natural Language Processing, EMNLP'08, pages 927-936, Stroudsburg, PA, USA. Association for Computational Linguistics.

[7] Michael Collins. 2003. *Head-Driven Statistical Models for Natural Language Parsing*. In Computational Linguistics 29, vol. 4, pages 589-637.

[8] Joao Silva, Luisa Coheur, Ana Cristina Mendes, and Andreas Wichert. 2011. *From symbolic to sub-symbolic information in question classification*. In Artificial Intellelligence Review, 35, pages 137-154.
