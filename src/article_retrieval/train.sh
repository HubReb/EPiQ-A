#!/bin/sh

echo "Merging natural questions training and dev set articles,,,"
python fuse.py
echo "Creating inverted index... This step may take several hours (depending on your hardware)"
python inverted_index.py
cd ../data_generation/
echo "Creating the merged dataset by fusing different versions of one article into a single article..."
python merge_article_versions.py
cd -
echo "Training the models... This may take several minutes..."
python train_models.py
