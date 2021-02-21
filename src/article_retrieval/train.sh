#!/bin/sh

python fuse.py
python inverted_index.py
cd ../data_generation/
python merge_article_versions.py
cd -
python train_models.py
