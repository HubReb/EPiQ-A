# Anwser extraction

There are 2 main tasks to extract accurate answers to preoprocessed questions in the retrieved articles: 
1) retrieve informative/relevant paragraphs in the article, and 2) extract the exact answer span from those paragraphs. 


### Extract the relevant paragraphs
We use Okapi BM25 ([gensim.summarization.bm25](https://radimrehurek.com/gensim_3.8.3/summarization/bm25.html))

### Extract the extact answer span
We use 2 pre-trained models from transformers will tokenize the paragraph:
* distilbert-base-uncased-squad2, or
* bert-large-uncased-whole-word-masking-finetuned-squad

We use the models of transformers(fine-tuned checkpoints): 
* transformers.DistilBertForQuestionAnswering, or
* transformers.BertForQuestionAnswering

1. Data Preprocessing
* To merge the wikipedia articles of the Natural Question train and dev set as the training corpus(so that the BM25 can train on it).
* Load the merge csv data into a dataframe and add the apreprocess the wikipedia articles to an extra column 'Text_Proc'.
<br>Run:
```
python3 data_preprocessing.py
```


(TODO)
This model can answer "yes/no questions" with a sentence instead of "yes/no" answers, thus we might also solve this problem.
