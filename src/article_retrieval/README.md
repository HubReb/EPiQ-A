# Article Retrieval

## Data

The code is only tested on the Natural Question dataset contained in the data folder.

## Model training

All training steps are contained in train.sh. Simply run
```
bash train.sh
```

to train one ranking model with Okapi BM 25 weights [1] (using gensim's summarization.bm25 module [2]) and one with TFIDF weights (using sklearn.feature_extraction.text.TfidfVectorizer class) and ranking according to cosine similarity.

## Model testing

There is only one evaluation file at the moment. Simply run
```
python evaluate_baseline.py
```
to evaluate the trained models.


## external code

All adapted, external code is marked in the files utilizing it. External packages are already included in the requirements.txt file.



## References
[1] Robertson, Stephen, and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Now Publishers Inc, 2009.
    https://www.researchgate.net/profile/Hugo_Zaragoza/publication/220613776_The_Probabilistic_Relevance_Framework_BM25_and_Beyond/links/53d9ce800cf2a19eee8807e8.pdf
    
[2] https://radimrehurek.com/gensim_3.8.3/summarization/bm25.html
