from collections import defaultdict
import pandas as pd
import re
import pickle
import os
from tqdm import tqdm
from gensim import corpora
from gensim.utils import simple_preprocess
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


class Articles():
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self._preprocessToDF()
        self.doc_list, self.words, self.corpus = self._createCorpus()

    def _preprocessToDF(self, urls=True):
        texts = []
        titles = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            url = f.readline()
            while True:
                if not url:
                    break
                if url[:3] == 'URL':
                    if not urls:
                        titles.append(url.strip().split('/')[-1].split('.')[0].replace('-', ' '))
                    else:
                        # Approximate titles by url
                        titles.append(url.strip().split()[-1])
                    line = f.readline()
                    currtext = ''
                    while line and line[:3] != 'URL':
                        if line == '\n':
                            line = f.readline()
                            continue
                        currtext += line.replace('\n', ' ')
                        line = f.readline()

                    texts.append(currtext.strip())
                    url = line
        dd = defaultdict(list)
        for i in range(len(titles)):
            dd['title'].append(titles[i])
            dd['text'].append(texts[i])
        df = pd.DataFrame(dd)
        # drop rows with no text
        # Need to alter this if using a different dataset
        df.drop([3978, 7096, 7108, 8869], inplace=True)
        return df

    def createNLPPipeline(self):
        def lemmatizer(doc):
            # This takesself, doc of tokens from the NER and lemmatizes them.
            # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so rem ove those.
            doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
            doc = u' '.join(doc)
            return nlp.make_doc(doc)

        def remove_stopwords(doc):
            # This will removeself, ords and punctuation.
            # Use token.text to return strings, which we'll need for Gensim.
            doc = [token.text for token in doc if not token.is_stop and not token.is_punct]
            return doc

        nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser'])

        stop_list = ["mrs", "ms", "say", "'s", "mr", '\ufeff1']
        nlp.Defaults.stop_words.update(stop_list)

        for word in STOP_WORDS:
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

        # The add_pipe function appends our functions to the default pipeline.
        nlp.add_pipe(lemmatizer, name='lemmatizer', after='ner')
        nlp.add_pipe(remove_stopwords, name="stopwords", last=True)
        return nlp

