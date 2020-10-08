import numpy as np
import pandas as pd
import pickle
from pprint import pprint
import matplotlib.pyplot as plt
from collections import defaultdict
from gensim.models import LdaModel, CoherenceModel
from joblib import dump, load
from wordcloud import WordCloud
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import os


class TopicModel():
    def __init__(self, num_topics, doc_list, words, corpus, passes=20, iterations=400, pretrainedfile=None):
        self.num_topics = num_topics
        self.corpus = corpus
        self.words = words
        self.passes = passes
        self.doc_list = doc_list
        self.iterations = iterations
        # intialize id2word
        temp = self.words[0]
        id2word = self.words.id2token
        if pretrainedfile:
            self.model = LdaModel.load(pretrainedfile)
        else:
            self.model = LdaModel(corpus=self.corpus, id2word=id2word, 
                    num_topics=self.num_topics, random_state=42,
                    update_every=1, passes=self.passes, 
                    iterations=self.iterations)

    def save(self, filename):
        dump(self, filename)

    @classmethod
    def load(cls, filename):
        return load(filename)

    def generateWordCloud(self, topic):
        fig, ax = plt.subplots()
        ax.imshow(WordCloud().fit_words(dict(self.model.show_topic(topic, 200))))
        ax.axis("off")
        ax.set_title("Topic #" + str(topic))
        return fig, ax

    def getDocLlist(self):
        return self.doc_list 

    def getWords(self):
        return self.words

    def getCorpus(self):    
        return self.corpus 

    def printTopics(self):
        pprint(self.model.print_topics())  

    def assignTopics(self, df):
        df['topic'] = [self._getSingleTopic(i) for i in range(df.shape[0])]

    def _getSingleTopic(self, i):
        res = self.model[self.corpus[i]]
        res.sort(key=lambda x: -x[1])
        return res[0][0]

    def getLogPerplexity(self):
        return self.model.log_perplexity()

    def getCoherence(self, kind='c_v'):
        if kind == 'c_v':
            coherence_model_lda = CoherenceModel(model=self.model, texts=self.doc_list, dictionary=self.words, coherence='c_v')
        elif kind == 'u_mass':
            coherence_model_lda = CoherenceModel(model=self.model, corpus=self.corpus, coherence='u_mass')
        else:
            raise ValueError("Only c_v and u_mass are currently supported")
        return coherence_model_lda.get_coherence()



