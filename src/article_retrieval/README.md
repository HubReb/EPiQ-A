# Article Retrieval

### Dataset

During training we fuse the articles of the Natural Questions dataset into one csv file: The development set contains  articles that are not present in the training set. The models can only retrieve documents from the document collectiot that are contained in the document collection, thus we have to fuse the two datasets.
Currently, the code is only evaluated on the Natural Question dataset.


### Construction of the inverted index

We roughly follow the approach outlined in [2], Chapter 2.2). All documents are tokenized and each word is lemmatized.  We then remove stop words and construct an inverted index from the remaining words. The index maps each word to a list of documents that contain the term. We opted for lemmatization instead of stemming as both approaches achieve about the same results for English. Furthermore, we do not need to add a new dependency. As such, tokenization and lemmatization both use the [spacy library](https://spacy.io/).

### Model training

We use two bag-of-words ranking models:
* Okapi BM25 [1] (using [gensim's summarization.bm25 module](https://radimrehurek.com/gensim_3.8.3/summarization/bm25.html))
*TFIDF (using [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)) and ranking according to cosine similarity. Please see their corresponding documentations for further details as weuse the packages default hyperparameters, except for the TfidfVectorizer's minimum document frequency parameter (min_df).
Note that the first time you train a model will take significantly longer. We apply tokenization, lemmatisation and stop word removal to the document collection and save the processed data to disk. This takes several hours. All further training runs will access this data and thus only take a few minutes to complete. 


The construction of the inverted index and all training steps are contained in train.sh. Simply run
```
bash train.sh
```

*Warning* The training takes about 20 GB RAM.

## Model testing

We only evaluate on the Natural Question dataset at the moment. We calculate both Mean Reciprocal Rank and R-precision.
To repeat the evaluation simply run
```
python evaluate_baseline.py
```
to evaluate the trained models.


## external code

All adapted, external code is marked in the files utilizing it. External packages are already included in the requirements.txt file.



## References
[1] Robertson, Stephen, and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Now Publishers Inc, 2009. https://www.researchgate.net/profile/Hugo_Zaragoza/publication/220613776_The_Probabilistic_Relevance_Framework_BM25_and_Beyond/links/53d9ce800cf2a19eee8807e8.pdf
    
[2] Christopher D. Manning, Prabhakar Raghavan and Hch Schütze, Introduction to Information Retrieval, Cambridge University Press. 2008.
