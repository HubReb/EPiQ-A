# Article Retrieval

## Data

The code is only tested on the Natural Question dataset contained in the data folder.

## Model training

All training steps are contained in train.sh. Simply run
```
bash train.sh
```

to train one ranking model with Opaki BM 25 weights [1] (using gensim.summarization.bm25 module) and one with TFIDF weights (using sklearn.feature_extraction.text.TfidfVectorizer class) and ranking according to cosine similarity.

## Model testing

There is only one evaluation file at the moment. Simply run
```
python test_baseline.py
```
to evaluate the trained models.




## References
[1] Robertson, Stephen; Zaragoza, Hugo (2009).  The Probabilistic Relevance Framework: BM25 and Beyond,
    http://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf
