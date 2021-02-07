import pandas as pd
import csv
import pickle
import codecs
import sys

def load_csv(path: str):
    '''
    :param path: path of the csv file
    :return dataframe:
    '''

    # Local
    # df = pd.read_csv(path, engine='python', error_bad_lines=False, quoting=csv.QUOTE_NONE) #encoding="utf8", quoting=csv.QUOTE_NONE

    # In Last
    df = pd.read_csv(path)

    return df

def merge_train_dev_articles(path1, path2, path_out):
    '''Read 2 csv files with the same header and merge them into one dataframe
    :param str: path of the csv file
    :return dataframe: merged dataframe
    '''
    df1 = load_csv(path1)
    df2 = load_csv(path2)

    # Merge on all common columns
    df = pd.merge(df1, df2,  on=list(set(df1.columns) & set(df2.columns)), how='outer')
    df.to_csv(path_out, index=False)


def merge_csvs(path1, path2, out_path):
    print('Merging training & dev set...')
    reader1 = csv.DictReader(open(path1))
    reader2 = csv.DictReader(open(path2))
    header = reader.fieldnames
    with open(out_path, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writerows(reader1)
        writer.writerows(reader2)
    print('Finished merging.')

def load_from_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

#### WORKING ON IT
def csv_to_list_of_passages(path):
    # To escape _csv.Error: field larger than field limit (131072)
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
            csv.field_size_limit(maxInt)

    # Start to split into passages
    all_passages = []
    articles = ''
    with open(path, 'rb') as fr: #, newline=''
        reader = csv.reader(codecs.iterdecode(fr, 'utf-8'))  # if rb mode
        header = reader.__next__()
        text_proc_idx = header.index('Text_Proc')
        while True:
            try:
                line = reader.__next__()
                if len(articles) == 0:
                    articles += line[text_proc_idx]
                else:
                    articles = articles + '\n\n' + line[text_proc_idx]
            except StopIteration:
                break
        passages = [passage.replace('\n', '').strip() for passage in
                    articles.split('\n\n') if passage.strip() != ""]  # a list of passages of the articles
        # print(passages)
        # print(len(passages))
    return passages