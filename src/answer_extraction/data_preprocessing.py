import spacy
from tqdm import tqdm
from data_utils import merge_train_dev_articles, load_csv
import re
import html
from subprocess import call
import csv
import codecs

# Preprocess Text(articles) in dataframe
class Data():
    def __init__(self, dataframe, model='en_core_web_sm'):
        # catch error if spaCy model is not available, installs it and restarts the script
        self.df = dataframe

        # Local
        # self.fp = f'./data/processed_article_corpus.csv'
        # Last
        self.fp = f'./processed_merged_wiki_text.csv'

        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f'spaCy model:\t{model} is not installed.\nInstalling now...')
            call(['python3', '-m', 'spacy', 'download', model])  # for terminal
            print('Restarting script...')
            os.execl(sys.executable, sys.executable, sys.argv[0])

        spacy.info()
        print(f'Dataframe being processed...')

    def remove_urls(self, text: str) -> str:
        """Removes urls from string

        :param text: text to be preprocessed
        :return str: text with urls removed
        """
        url_pattern = re.compile('(\w+:\/\/\S+)|(www\.[^\s]*[\s]*)')
        text = re.sub(url_pattern, ' ', text)
        return text

    def remove_some_punctuations(self, text: str) -> str:
        punctuations = '\"`\'#|*{}[]'

        for punct in punctuations:
            text = text.replace(punct, ' ')

        return text

    def remove_nonascii(self, text: str) -> str:
        """Removes all non-ASCII characters from text

        :param text: text to be preprocessed
        :return str: text with non-ASCII characters removed
        """
        return text.encode('ascii', 'ignore').decode('ascii')

    def escape_html(self, text: str) -> str:
        s = html.escape("""& < " ' >""").split()  # s = ["&amp;", "&lt;", "&quot;", "&#x27;", "&gt;"]
        for e in s:
            text = text.replace(e, ' ').strip()
        return text

    def remove_multiple_whitespaces(self, text: str) -> str:
        """Removes mutiple whitespaces from a text

        :param text: text to be preprocessed
        :return str: text with all and multiple whitespaces removed
        """
        whitespaces_pattern = re.compile(r' +')

        return re.sub(whitespaces_pattern, ' ', text).strip()

    def preprocess(self, lowercase: bool = True):
        """Complete preprocessing pipeline of underlying dataframe

        :Keyword Argument lowercase: if Ture => lowercasing text (default: {True})
        :return str: text with emojis removed
        """
        tqdm.pandas(desc='Processing data', ncols=1000)

        if lowercase:
            self.df['Text_Proc'] = self.df['Text'].str.lower()

        self.df['Text_Proc'] = self.df['Text_Proc'].astype(str)
        self.df['Text_Proc'] = self.df['Text_Proc'].progress_apply(self.remove_nonascii)
        self.df['Text_Proc'] = self.df['Text_Proc'].progress_apply(self.remove_urls)
        self.df['Text_Proc'] = self.df['Text_Proc'].progress_apply(self.escape_html)
        self.df['Text_Proc'] = self.df['Text_Proc'].progress_apply(self.remove_some_punctuations)
        self.df['Text_Proc'] = self.df['Text_Proc'].progress_apply(self.remove_multiple_whitespaces)

    def main(self):
        """Preprocessing and saving processed .csv table
        """
        self.preprocess(lowercase=True)
        # Only sace column 'Wikipedia_ID' and 'Text_Proc'
        selected_dataframe = self.df[['Wikipedia_ID', 'Text_Proc']]
        selected_dataframe.to_csv(self.fp, index=False) #
        # self.df.to_csv(self.fp, index=False)
        print(f'\nData frame written to {self.fp}')

if __name__ == '__main__':
    # Local
    # train_wiki_text_file = "./data/nq_train_wiki_text_short.csv"
    # dev_wiki_text_file = "./data/nq_dev_wiki_text_short.csv"
    # merged_path = f'./data/nq_merged_wiki_text.csv'

    # Last:
    # train_wiki_text_file = "/proj/mahoni/project/data/nq_train_wiki_text.csv"
    # dev_wiki_text_file = "/proj/mahoni/project/data/nq_dev_wiki_text.csv"
    # merged_path = f'./nq_merged_wiki_text.csv'
    # merge_train_dev_articles(train_wiki_text_file, dev_wiki_text_file, merged_path)

    # df_train_dev = load_csv(merged_path)


    # Local
    # df_train_dev = load_csv("./data/nq_dev_train_wiki_text_merged_short.csv")
    # Last
    df_train_dev = load_csv("/proj/epiqa/EPiQ-A/data/article_retrieval/nq_dev_train_wiki_text_merged.csv")

    preprocess_corpus = Data(df_train_dev)
    preprocess_corpus.main()