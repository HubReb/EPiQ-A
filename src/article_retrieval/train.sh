#!/bin/sh

python fuse_sets_and_create_index.py
python split_train_dataset.py
python train_models.py
