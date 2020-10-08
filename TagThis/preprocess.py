import pandas as pd
import re
from gensim import corpora
from gensim.utils import simple_preprocess
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import os
import pickle
from tqdm import tqdm
from collections import defaultdict

FILENAME = 'nytimes_news_articles.txt'
basepath = os.path.dirname(__file__)
filepath = os.path.abspath(os.path.join(basepath, '..', 'data', FILENAME))


def preprocessToDF(urls=True):
    texts = []
    titles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        url = f.readline()
        while True:
            if not url:
                break
            if url[:3] == 'URL':
                if not urls:
                    titles.append(url.strip().split('/')[-1].split('.')[0].replace('-', ' '))
                else:
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
    df.drop([3978, 7096, 7108, 8869], inplace=True)
    return df


def createCorpus():
    if os.path.exists('processedtext.pkl'):
        with open('processedtext.pkl', 'rb') as f:
            doc_list, words, corpus = pickle.load(f)

    else:
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(simple_preprocess(str(sentence),deacc=True))

        df = preprocessToDF()                
        data = df.text.values.tolist()
        # remove city information
        for i in range(len(data)):
            if data[i].split()[1] == '—':
                data[i] = ' '.join(data[i].split()[2:])
        # remove punctuation
        data = [re.sub('[,\.!?]', '', x) for x in data]
        # lowercase everything
        data = [x.lower() for x in data]
        data_words = list(sent_to_words(data))

        nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser'])

        stop_list = ["mrs", "ms", "say", "'s", "mr", '\ufeff1']
        nlp.Defaults.stop_words.update(stop_list)

        for word in STOP_WORDS:
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

        def lemmatizer(doc):
            # This takes in a doc of tokens from the NER and lemmatizes them. 
            # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so remove those.
            doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
            doc = u' '.join(doc)
            return nlp.make_doc(doc)
            
        def remove_stopwords(doc):
            # This will remove stopwords and punctuation.
            # Use token.text to return strings, which we'll need for Gensim.
            doc = [token.text for token in doc if not token.is_stop and not token.is_punct]
            return doc

        # The add_pipe function appends our functions to the default pipeline.
        nlp.add_pipe(lemmatizer, name='lemmatizer', after='ner')
        nlp.add_pipe(remove_stopwords, name="stopwords", last=True)

        newest_doc = data
        doc_list = []
        # Iterates through each article in the corpus.
        for doc in tqdm(newest_doc):
            # Passes that article through the pipeline and adds to a new list.
            pr = nlp(doc)
            doc_list.append(pr)

        # Creates, which is a mapping of word IDs to words.
        words = corpora.Dictionary(doc_list)

        # Turns each document into a bag of words.
        corpus = [words.doc2bow(doc) for doc in doc_list]

        with open('processedtext.pkl', 'wb') as f:
            pickle.dump([doc_list, words, corpus], f)
    return doc_list, words, corpus

def preprocessSingleInput(inputDoc):
    # assumes inputDoc is a string of sentences
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(simple_preprocess(str(sentence),deacc=True))
            
    data = inputDoc 
    if len(data) < 10:
        raise ValueError('Please post an article with more than a 10 words')
    # remove city information
    if data.split()[1] == '—':
        data = ' '.join(data.split()[2:])
    # remove punctuation
    data = re.sub('[,\.!?]', '', data) 
    # lowercase everything
    data = data.lower() 
    data_words = list(sent_to_words(data))
    nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser'])

    # Updates spaCy's default stop words list with my additional words. 
    stop_list = ["mrs", "ms", "say", "'s", "mr", '\ufeff1']
    nlp.Defaults.stop_words.update(stop_list)

    # Iterates over the words in the stop words list and resets the "is_stop" flag.
    for word in STOP_WORDS:
        lexeme = nlp.vocab[word]
        lexeme.is_stop = True

    def lemmatizer(doc):
        # This takes in a doc of tokens from the NER and lemmatizes them. 
        # Pronouns (like "I" and "you" get lemmatized to '-PRON-', so remove those.
        doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
        doc = u' '.join(doc)
        return nlp.make_doc(doc)
        
    def remove_stopwords(doc):
        # This will remove stopwords and punctuation.
        # Use token.text to return strings, which we'll need for Gensim.
        doc = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return doc

    # The add_pipe function appends our functions to the default pipeline.
    nlp.add_pipe(lemmatizer, name='lemmatizer', after='ner')
    nlp.add_pipe(remove_stopwords, name="stopwords", last=True)

    return nlp(data)

if __name__ == '__main__':
    df = preprocessToDF()
    doc_list, words, corpus = createCorpus()
    print(df.head())


