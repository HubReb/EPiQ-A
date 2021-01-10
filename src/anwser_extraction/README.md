# Anwser extraction

There are two main tasks to extract accurate answers to preoprocessed questions in the retrieved articles: 
1) retrieve informative/relevant paragraphs in the article, and 2) extract the exact answer span from those paragraphs. 


### Relevant paragraphs extraction
We use Okapi BM25 ([gensim.summarization.bm25](https://radimrehurek.com/gensim_3.8.3/summarization/bm25.html))

### Extract the extact answer span
2 pre-trained models from transformers will tokenize the paragraph:
* distilbert-base-uncased-squad2, or
* bert-large-uncased-whole-word-masking-finetuned-squad

We use the from transformers provided models(fine-tuned checkpoints): 
* transformers.DistilBertForQuestionAnswering, or
* transformers.BertForQuestionAnswering



(TODO)
This model can answer "yes/no questions" with a sentence instead of "yes/no" answers, thus we might also solve this problem.
