#!/usr/bin/env python3

""" Merging of articles to have all article version in one article without duplicated paragraphs """

import re
import sys
import csv


def merge_article_versions(filename: str) -> None:
    """
    Merges all wikipedia articles in the given csv file
    that have the same title.
    """
    old_max_filed_size = csv.field_size_limit()
    csv.field_size_limit(sys.maxsize)
    articles = dict()

    # First, we read all articles and bucket them by title
    # The title is always in the first line of the article text
    with open(filename) as wf:
        for _, link, text in csv.reader(wf):
            if link.strip() == "Wikipedia_ID":
                continue
            title = text.split('\n')[0].strip().lower()
            title = re.sub('- wikipedia', '', title).strip()

            texts = text.split('\n\n')

            if title in articles:
                articles[title]['links'].append(link)
                articles[title]['texts'].extend(texts)

            else:
                articles[title] = {'links': [link], 'texts': texts}

    # Next, we create a new file containing all merged articles
    # We remove duplicate paragraphs and save wikipedia links
    # of all articles that got merged
    name, _ = filename.split('.csv')
    with open(name + '_merged' + '.csv', 'w') as wf:
        # Write header
        csv_writer = csv.writer(wf, delimiter=',')
        csv_writer.writerow(['', "Wikipedia_ID", "Text"])

        deleted_paragraph_counter = 0

        for index, (title, article_info) in enumerate(articles.items()):
            print("Processing article {}/{}\tNum deleted paragraphs: {}"
                  .format(index+1, len(articles), deleted_paragraph_counter), end='\r')
            links = article_info['links']
            texts = article_info['texts']

            keep_paragraphs = []
            for paragraph in texts:
                # Within a paragraph, we can ignore all other types
                # of whitespace
                paragraph = re.sub(r"\s", " ", paragraph)
                paragraph = paragraph.lower().strip()

                # Remove empty paragraphs
                if not paragraph:
                    continue

                # Compare current paragraph to all previous paragraphs
                # of the same merged article. Only keep unseen paragraphs.
                keep = True
                for keep_paragraph in keep_paragraphs:
                    if hash(paragraph) == hash(keep_paragraph):
                        keep = False
                        deleted_paragraph_counter += 1
                        break

                if keep:
                    keep_paragraphs.append(paragraph)

            if len(keep_paragraphs) >= 1:
                text = "\n\n".join(keep_paragraphs)
                print(text)
                csv_writer.writerow([str(index), " ".join(links), text])

    csv.field_size_limit(old_max_filed_size)


if __name__ == '__main__':
    merge_article_versions("../../data//article_retrieval/nq_dev_train_wiki_text.csv")
