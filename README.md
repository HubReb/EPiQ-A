# EPiQ-A

## Team members
* Marinco Möbius (moebius@cl.uni-heidelberg.de)
* Jin Huang (huang@cl.uni-heidelberg.de)
* Leander Girrbach (girrbach@cl.uni-heidelberg.de)
* Rebekka Hubert (hubert@cl.uni-heidelberg.de)

## Existing code
Pre-existing code is clearly marked in both the README and the scripts. Note that both the data generation and article retrieval have their own READMEs due to the complexity of these modules.

## Requirements

All requirements are given in requirements.txt.

## Installation

First install the requirements. This steps varies depending on your own python setup.

### virtual environments 

Run `pip install -r requirements.txt`

### Anaconda

Run ` conda install --file requirements.txt`.


Then, change into the `src` folder  and run `pip install .`. If you intend to continue development, run `pip install . --editable` to avoid running `pip install .` after each change to the code.

## Usage Guide
### Data Preparation

The data preparation requires several manual steps, because the NaturalQuestions dataset requires you to log into your Google account to acquire it. Please change into the `src/data_generation` folder and follow the steps detailed in the README there to run the data preparation. 
Note that the data preparation may take several hours  to complete and requires several GB of RAM.

### Model Training
 
All training steps are given in the `src/train.sh` bash script. Simply run `bash train.sh`. Note that this steps takes several hours to complete if you run it for the first time.

### Running Experiments
 1. Navigate into the `src/` folder
 2. Run `python main.py --mode evaluate`. Other options are `--mode interactive` for asking your own questions in an interactive shell and `--mode evaluate_article_retrieval` for evaluating only the article retrieval module.
 3. Our implementation supports multiple modes of retrieving relevant articles and paragraphs, as well as configuring many important parameters. Run `python main.py --help` for an overview.

Note that, in total, our implementation requires ~30 GB of RAM to load all trained models and the data.


## Future Planning

At the moment our article retrieval steps serves to weaken our overall approach. For instance, the Recall@10 is only about 50\% and our Precision@k performance is disturbingly low. We believe that replacing our TF-IDf approach with a neural retrieval model would significantly improve this pipeline step's results and thus help us achieve better overall performance. 
Another possible step would be to cut down on the number of retrieved articles. Right now, the user decides how many articles are retrieved and passed on to the next pipeline step with a default value of ten. It may be feasible to take a closer look at the scoring of the articles and discard all articles with a similarity value below a threshold. However, one would first need to think of an adequate way to determine the value of this threshold as most questions have a low similarity to all possible articles and hard-coding such a threshold would result in the module simply returning no articles at all.


### High-level Architecture Description:
Our project requires 2 main parts: Data preprocessing and the question answering pipeline. The pipeline consists of 4 different parts: 
 1. Preprocessing the query
 2. retrieval of articles that may contain the answer to the query
 3. the retrieval of paragraphs therein
 4. eventually the extraction of the answer from the retrieved articles.

Retrieval of the articles requires its own submodule  due to the different, interchangeable components in this step. All other functionality is organised in single files in the `src`-folder.

In the following, we describe each pipeline step in detail.

#### Question Processing
The question processing component applies fundamental prepocessing to the given natural language question (represented as string). This includes tokenisation, and optionally stop-word-removal, lowercasing, and lemmatising or stemming. Lemmatising and stemming are mutually exclusive.

For tokenisation and lemmatising, we use the [spacy library](https://spacy.io/) [1]. For stop-word-removal and stemming, we use [NLTK](https://www.nltk.org/) [3].


#### Article Retrieval


We first build an inverted index from the articles in the entire dataset: We lemmatise each tokenised word in the dataset. Then we remove stop words. The remaining words are used to build the index. Each word maps to a list of documents that contain this word.

For querying the document collection, the words in the lemmatised query are mapped to the articles via the inverted index and the retrieved articles are ranked with respect to the query. Currently, we have two ranking, bag-of-words methods implemented: The TF-IDF Weighting introduced in the lecture with cosine similarity to the TF-IDF vector of the query for ranking and Okapi BM25. Note that our current implementation of TF-IDF uses [sklearns](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) default parameters.
For the Okapi BM25 we use the implementation provided by the python package [gensim](https://radimrehurek.com/gensim_3.8.3/summarization/bm25.html).
Regardless of the chosen ranking model we pass the top ten ranked articles to the answer extraction component.
All further detail is given in the sub-modules folder `article_retrieval`.


#### Paragraph retrieval
To extract relevant paragraphs from the retrieved articles, we again use a BM25 model (gensim implementation). 

The model first maps retrieved articles to their paragraphs. The paragraphs are cached. Then, the BM25-model ranks these paragraphs according to the given query. We return the top ranked paragraphs. Because the pretrained answer extraction models (see below) can only handle contexts of limited length, we split each paragraph into (overlapping) context windows.

Note the two stage process of extracting relevant passages: First, we extract articles, then, from the retrieved articles, we retrieve paragraphs.


#### Answer extraction
For question answering, the [huggingface transformers library](https://github.com/huggingface/transformers) [4] provides pretained models that can extract answer spans to given questions from given contexts. To use these models, we make use of the library's `pipeline`-API ( https://huggingface.co/transformers/main_classes/pipelines.html ).

For our experiments, we use the `distilbert-base-uncased-distilled-squad` and `roberta-base-squad2` models, but our script, via the `pipeline`-API, allows specifying any other valid model.


### Experiments

#### Evaluation of Article Retrieval

Because our dataset contains only one correct article for each question, we focus on the rank the module assigns to the correct document and whether this document is passed on the the next pipeline step: We calculate R-precision, Recall@k, Precision@k and also compute Mean Reciprocal Rank (MMR).

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

These results show that we may still improve upon our current method. Note that surprisingly the TF-IDF weighting ranking method achieves far better results.

#### Evaluation of the full model

Evaluating question answering is difficult, because potentially many different concrete strings represent the correct answer. Furthermore, to some questions, no unique correct answer exists.

For our evaluation, we still focus on metrics that measure string overlap of the correct answer as provided by the dataset and the answer predicted by our model. To get a better impression of our model's performance, we measure:
 * F1 score (between gold tokens and predicted tokens)
 * Exact match (predicted answer must match gold answer exactly)
 * BLEU score (between gold tokens and predicted tokens)
 * Word-Error-Rate (normalised Levenshtein-Edit-Distance)
 * Jaccard-Index of types that appear in correct and predicted answer

First we report results for the default pretrained model (`distilbert-base-uncased-distilled-squad`):

|                | F1   | Exact Match | BLEU  | WER  | Jaccard |
|----------------|------|-------------|-------|------|---------|
| TF-IDF         | 0.01 | 0.00        | 15.41 | 1.37 | 0.00    |
| TF-IDF + Index | 0.01 | 0.00        | 15.13 | 1.40 | 0.00    |
| BM25           | 0.01 | 0.00        | 15.21 | 1.39 | 0.00    |
| BM25 + Index   | 0.01 | 0.00        | 15.21 | 1.39 | 0.00    |
| No retrieval   | 0.12 | 0.08        | 18.83 | 1.23 | 0.11    |


We can see that all retrieval methods perform equally bad. When not using any article retrieval but retrieving paragraphs directly from the set of all paragraphs in the dataset, performance improves considerably, but remains low in absolute terms.

The same observation can be made for another pretrained model (`roberta-base-squad2`):

|                | F1   | Exact Match | BLEU  | WER  | Jaccard |
|----------------|------|-------------|-------|------|---------|
| TF-IDF + Index | 0.01 | 0.00        | 16.01 | 1.27 | 0.00    |
| No retrieval   | 0.15 | 0.09        | 19.72 | 1.18 | 0.13    |

The low performance of the model with article retrieval cannot be explained fully by insufficient quality of the retrieval component, because we observe Recall@10 of up to 0.5 for the article retrieval module (TF-IDF weighting).

Therefore, it rather seems to be the case that errors accumulate between the multiple submodules. This confirms the current trend to engineer question answering systems that work with as few components as possible.


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

[1] Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. *spaCy: Industrial-strength Natural Language Processing in Python*. Zenondo.

[2] Radim Rehurek and Petr Sojka. 2010. *Software Framework for Topic Modelling with Large Corpora*. In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks, pages 45--50.

[3] Bird, Steven, Ewan Klein, and Edward Loper (2009), Natural Language
Processing with Python, O'Reilly Media.

[4] Wolf, Thomas, et al. "Transformers: State-of-the-art natural language processing." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. 2020.

