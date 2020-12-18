from os.path import join
from argparse import ArgumentParser
from collections import OrderedDict

import json
import html2text
import pandas as pd
from tqdm import tqdm


def has_short_answer(annotation):
    return len(annotation["short_answers"]) > 0


def main():
    parser = ArgumentParser(description='Preprocess Natural Question dataset.')
    parser.add_argument('-subset',
                        default='train',
                        dest='subset',
                        help='Set processing of train/dev-set.')
    args = parser.parse_args()
    path = f"../../data/QA/NaturalQuestions/{args.subset}_set/"
    subsets = {
        'dev': "v1.0-simplified_nq-dev-all.jsonl",
        'train': "v1.0-simplified_simplified-nq-train.jsonl"
    }

    filename = join(path, subsets[args.subset])
    sections_to_skip = {
        "Retrieved from ``",
        "##  References",
        "##  Footnotes",
        "hidden categories :",
        "##  Notes",
        "##  See also"
        }
    dev = args.subset == 'dev'
    totals = {'train': 307373, 'dev': 7830}
    pbar = tqdm(total=totals[args.subset])
    nat_data = pd.DataFrame(columns=['Question', 'Answer', 'Wikipedia_ID'])
    url_to_text = pd.DataFrame(columns=['Wikipedia_ID', 'Text'])
    with open(filename) as f:
        for line in f:
            data = json.loads(line, object_pairs_hook=OrderedDict)
            question = data["question_text"]
            if dev:
                tokens = [token_info['token'] for token_info in data['document_tokens']]
            wp_id = data["document_url"]
            annotation = data['annotations'][0]
            yes_no = annotation['yes_no_answer']
            if yes_no == 'NONE':
                if has_short_answer(annotation):
                    answers = annotation['short_answers']
                    start, end = answers[0]['start_token'], answers[0]['end_token']
                    if dev:
                        document_text = " ".join(tokens)
                        answer_text = " ".join(tokens[start:end])
                    else:
                        document_text = data['document_text']
                        answer_text = " ".join(document_text.split(" ")[start:end])
                else:
                    pbar.update(1)
                    continue
            else:
                # yes/no answer
                answer_text = yes_no.lower().capitalize()
            nat_entry = {
                'Question': question,
                'Answer': answer_text,
                'Wikipedia_ID': wp_id
            }
            nat_data = nat_data.append([nat_entry])
            # extract text
            if wp_id not in url_to_text['Wikipedia_ID']:
                # remove html tags
                cleaned_text = html2text.html2text(document_text)
                # remove unnecessary wikipedia sections
                for marker in sections_to_skip:
                    cleaned_text = cleaned_text.split(marker)[0]
                url2text_entry = {
                    'Wikipedia_ID': wp_id,
                    'Text': cleaned_text
                }
                url_to_text = url_to_text.append([url2text_entry])
            pbar.update(1)
    pbar.close()
    nat_data = nat_data.reset_index().drop(['index'], axis=1)
    nat_data.to_csv(join(path, f'nq_{args.subset}.csv'), index=False)
    url_to_text = url_to_text.reset_index().drop(['index'], axis=1)
    url_to_text = url_to_text.drop_duplicates()
    url_to_text.to_csv(join(path, f'nq_{args.subset}_wiki_text.csv'), index=False)


if __name__ == "__main__":
    main()
