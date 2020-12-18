#!/bin/sh

python fuse.py
python inverted_index.py
python split_train_dataset.py
python train_models.py
