import sys
from os import mkdir, remove
from os.path import join, exists
from argparse import ArgumentParser

import wget
import zipfile


# taken from https://stackoverflow.com/questions/58125279/python-wget-module-doesnt-show-progress-bar
def bar_progress(current, total, width=80):
    # shows progress bar for wget download
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def main():
    parser = ArgumentParser(description='Download dataset.')
    parser.add_argument('-dataset',
                        default='ConvQuestions',
                        dest='dataset',
                        help='Name the datset you want to download: currently only "ConvQuestions"!!!')

    args = parser.parse_args()
    datasets = {
        'ConvQuestions': [
            "http://qa.mpi-inf.mpg.de/convex/ConvQuestions_train.zip",
            "http://qa.mpi-inf.mpg.de/convex/ConvQuestions_dev.zip",
            "http://qa.mpi-inf.mpg.de/convex/ConvQuestions_test.zip"
            ],
        'NaturalQuestions': [
            "https://storage.cloud.google.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz",
            "https://storage.cloud.google.com/natural_questions/v1.0-simplified/nq-dev-all.jsonl.gz"
            ]
    }

    subsets = ['train', 'dev', 'test']
    if args.dataset in datasets:
        urls = datasets[args.dataset]
        if not exists(args.dataset):
            mkdir(args.dataset)
        for url, subset in zip(urls, subsets):
            print(f'downloading {subset} set from {url}.')
            wget.download(url, args.dataset, bar=bar_progress)
            zip_name = join(args.dataset, f"{args.dataset}_{subset}.zip")
            with zipfile.ZipFile(zip_name, 'r') as zip_file:
                zip_file.extractall(args.dataset)
            remove(zip_name)
    else:
        print('You specified an invalid dataset name: choose "ConvQuestions".')


if __name__ == "__main__":
    main()
