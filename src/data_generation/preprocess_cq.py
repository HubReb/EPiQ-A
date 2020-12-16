# Preprocessing for ConvQuestions (cq)
import re
import sys
import time

import json
import spacy
import pandas as pd
import urllib3
from tqdm import tqdm
from bs4 import BeautifulSoup
from SPARQLWrapper import SPARQLWrapper, JSON


# function adapted from https://query.wikidata.org/
def query_wikipedia_url_from_wikidata_id(wd_id, language="en"):
    r"""Returns associated wikipedia url of wikidata entity.

    Args:
      wd_id:    Identifier of wikidata entity, e.g. Q9013673
      language: Language code of wikipedia article

    Returns:
      String with URL of the article in the chosen language.
      None is returned if there is no associated wikipedia article.
    """

    def get_results(endpoint_url, query):
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()

    query = f"""SELECT ?lang ?name ?url
        WHERE {{
          ?url schema:about wd:{wd_id} ;
                   schema:inLanguage ?lang ;
                   schema:name ?name ;
                   schema:isPartOf [ wikibase:wikiGroup "wikipedia" ] .
          FILTER(?lang in ('{language}')) .
        }}"""

    endpoint_url = 'https://query.wikidata.org/sparql'
    results = get_results(endpoint_url, query)["results"]["bindings"]

    if results:
        return results[0]["url"]["value"]
    else:
        None


def load_page(url):
    urllib3.disable_warnings()
    http = urllib3.PoolManager()
    page = http.request('GET', url).data
    return page


def parse_page(page):
    soup = BeautifulSoup(page, 'lxml')
    return soup


def get_wiki_content(wp_id):
    r"""Returns content (infobox & text) of a Wikipedia article.

    Args:
      wp_id:    Url of Wikipedia article.

    Returns:
      String with content of the wikipedia article.
    """
    soup = parse_page(load_page(wp_id))
    # extract infobox
    infobox = list()
    infobox_text = ""
    info_box_search = soup.find('table', class_=re.compile(r'infobox \w+'))
    if info_box_search:
        for items in info_box_search.find_all('tr')[1::1]:
            data = items.find_all(['th', 'td'])
            for element in data:
                if element.text:
                    infobox.append(element.text.strip())
                elif element.a:
                    infobox.append(element.a.text.strip())
        infobox_text = "\n".join(infobox)
    # Extract the plain text content
    content = list()
    # we want all paragraphs, lists and headings
    tags = re.compile(r'p|ul|h\d+')
    # we only want the top-level tags -> recursive=False
    for tag in soup.find('div', {"class": "mw-parser-output"}).find_all(tags, recursive=False):
        content.append(tag.text)
    content_text = "\n".join(content)
    text = infobox_text + "\n" + content_text
    # Drop footnote superscripts in brackets
    text = re.sub(r'\[.*?\]+', '', text)
    return text


def main():
    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'textcat'])
    subsets = ['train', 'dev', 'test']
    for subset in subsets:
        print('subset:', subset)
        filename = f'../../data/QA/ConvQuestions/{subset}_set/{subset}_set_ALL.json'
        with open(filename) as f:
            file_content = f.read()
        conversations = json.loads(file_content)
        conv_data = pd.DataFrame(columns=['Question', 'Answer', 'Wikipedia_ID'])
        url_to_text = pd.DataFrame(columns=['Wikipedia_ID', 'Text'])
        for conversation in tqdm(conversations):
            # identifier for wikidata entities, e.g. Q9013673
            wd_id = conversation['seed_entity'].split('/')[-1].split('?')[0]
            # url of wikipedia article
            try:
                wp_id = query_wikipedia_url_from_wikidata_id(wd_id)
            except urllib3.exceptions.ProtocolError as e:
                print(e)
                print('retrying in 60 seconds.')
                time.sleep(61)
                wp_id = query_wikipedia_url_from_wikidata_id(wd_id)
            if not wp_id:
                # some wikidata pages don't have a link to wikipedia (e.g. https://www.wikidata.org/wiki/Q2721108)
                # we skip those cases
                continue
            question_info = conversation['questions'][0]
            question = question_info['question']
            answer = question_info['answer_text']
            conv_entry = {
                'Question': question,
                'Answer': answer,
                'Wikipedia_ID': wp_id
                }
            if wp_id not in url_to_text['Wikipedia_ID']:
                text = get_wiki_content(wp_id)
                tokenized_text = list()
                for line in text.split('\n'):
                    tokens = [token.text for token in nlp(line)]
                    tokenized_text.append(" ".join(tokens))
                tokenized_text = "\n".join(tokenized_text)
                conv_data = conv_data.append([conv_entry])
                url2text_entry = {
                                'Wikipedia_ID': wp_id,
                                'Text': tokenized_text
                            }
                url_to_text = url_to_text.append([url2text_entry])
        conv_data = conv_data.reset_index(drop=True)
        conv_data = conv_data.drop_duplicates()
        conv_data.to_csv(f'cq_new_{subset}_all.csv', index=False)
        url_to_text = url_to_text.reset_index(drop=True)
        url_to_text = url_to_text.drop_duplicates()
        url_to_text.to_csv(f'cq_new_{subset}_wiki_text.csv', index=False)


if __name__ == "__main__":
    main()
