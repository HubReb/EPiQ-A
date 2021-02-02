# Anwser extraction

TODO:
* Fix the function 'csv_to_list_of_passages()' in data_utils.py to
* Train BM25 Model
* Combinate the Reader and trained BM25 model
* Fine tuning the BM25 parameters on training set and test on dev set
* Evaluation<br><br>

There are 2 main tasks to extract accurate answers to preoprocessed questions in the retrieved articles: 
1) retrieve informative/relevant paragraphs in the article, and 2) extract the exact answer span from those paragraphs. 


### 1. Extract the relevant paragraphs
We use Okapi BM25 ([gensim.summarization.bm25](https://radimrehurek.com/gensim_3.8.3/summarization/bm25.html))

1. Data Preprocessing
* To merge the wikipedia articles of the Natural Question train AND dev set as the training corpus(so that the BM25 can train on it).
* Load the merge csv data into one dataframe and add the preprocess the wikipedia articles.
<br>Run:
```
python3 data_preprocessing.py
```
The preprocessed corpus will be saved in ./data/processed_article_corpus.csv with 2 columns: 'Wikipedia_ID','Text_Proc(preprocessed articles)<br>

2. Train the BM25 model (The Okapi BM25 class is in file 'passage_BM25.py')
* Train the model on the tprocessed_article_corpus.csv (on paragraphs level)
<br>Run:
```
python3 train_BM25.py
```
The trained BM25 model (will be) is saved in trained_bm25.pkl


### 2. Extract the extact answer span
We use 2 pre-trained models from transformers will tokenize the paragraph:
* distilbert-base-uncased-squad2, or
* bert-large-uncased-whole-word-masking-finetuned-squad

We use the models of transformers(fine-tuned checkpoints): 
* transformers.DistilBertForQuestionAnswering, or
* transformers.BertForQuestionAnswering

* (maybe)This model can answer "yes/no questions" with a sentence instead of "yes/no" answers, thus we might also solve this problem.<br>

Idea:
* Evaluation: Macro-averaged F1 score im SQuAD / Jaccard-Index on Token-Overlap
* plus one long answer from BM25 on sentence level?
