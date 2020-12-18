# Data

* [Conv Questions](https://convex.mpi-inf.mpg.de/)
* [Natural Questions](https://ai.google.com/research/NaturalQuestions/download)


## Download Data

### Conv Questions [1]
run [download_dataset.py](download_dataset.py)
```
python download_dataset.py
```


### Natural Questions [2]

Please download manually (google account needed) and put the data in the [NaturalQuestions](../../data/QA/NaturalQuestions) folder.  
**Note**: The test-data is not publicly available.

## Preprocess Data

### Conv Questions
```
python preprocess_nq.py
```

### Natural Questions
```
python download_dataset.py -subset "train"
python download_dataset.py -subset "dev"
```

## Data Analysis

see [Analysis-CQ-QuestionType&EntityType.ipynb](Analysis-CQ-QuestionType&EntityType.ipynb) for an analysis of the CQ dataset.

## References

[1] Christmann, P., Saha Roy, R., Abujabal, A., Singh, J., & Weikum, G. (2019, November). **Look before you hop: Conversational question answering over knowledge graphs using judicious context expansion**. In *Proceedings of the 28th ACM International Conference on Information and Knowledge Management* (pp. 729-738). [[URL]](https://doi.org/10.1145/3357384.3358016)

[2] Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., ... & Toutanova, K. (2019). **Natural questions: a benchmark for question answering research**. *Transactions of the Association for Computational Linguistics*, 7, 453-466. [[URL]](https://transacl.org/ojs/index.php/tacl/article/view/1455)