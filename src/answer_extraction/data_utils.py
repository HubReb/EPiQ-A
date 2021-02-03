import pandas as pd
import csv


def load_csv(path: str):
    '''
    :param path: path of the csv file
    :return dataframe:
    '''
    df = pd.read_csv(path)  # , engine='python', error_bad_lines=False , delimiter='\t', encoding="utf8", engine='python', error_bad_lines=False
    return df


def merge_train_dev_articles(path1, path2):
    '''Read 2 csv files with the same header and merge them into one dataframe
    :param str: path of the csv file
    :return dataframe: merged dataframe
    '''
    df1 = load_csv(path1)
    df2 = load_csv(path2)

    # Merge on all common columns
    df = pd.merge(df1, df2, on=list(set(df1.columns) & set(df2.columns)), how='outer')

    # df.to_csv("output.csv", index=False)
    return df
