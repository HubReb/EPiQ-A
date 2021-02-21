#!/bin/sh

python fuse.py
python inverted_index.py
cd ../data_generation/
python merge_article_versions.py
python split_train_dataset.py
python train_models.py
