from pprint import pprint
import matplotlib.pyplot as plt
from gensim.models import LdaModel, CoherenceModel
from joblib import dump, load
from wordcloud import WordCloud


class TopicModel():
    def __init__(self, news, num_topics, passes=20, iterations=400, pretrainedfile=None):
        self.news = news
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        # intialize id2word
        temp = self.news.words[0]
        id2word = self.news.words.id2token
        if pretrainedfile:
            self.model = LdaModel.load(pretrainedfile)
        else:
            self.model = LdaModel(corpus=self.news.corpus, id2word=id2word, 
                    num_topics=self.num_topics, random_state=42,
                    update_every=1, passes=self.passes, 
                    iterations=self.iterations)
        self._assignTopics()

    def __str__(self):
        return "LDA model with " + str(self.num_topics) + " topics"
    
    def __repr__(self):
        return str(self)

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

    def saveAllWorldClouds(self):
        for i in range(self.model.num_topics):
            fig, ax = plt.subplots()
            ax.imshow(WordCloud().fit_words(dict(self.model.show_topic(i, 200))))
            ax.axis("off")
            ax.set_title("Topic #" + str(i))
            plt.savefig('images/LDATopic' + str(i) + '.jpg')
        return

    def printTopics(self):
        pprint(self.model.print_topics())

    def _assignTopics(self):
        if 'topic' not in self.news.df.columns:
            self.news.df['topic'] = [self._getSingleTopic(i) for i in range(self.news.df.shape[0])]
            return
        else:
            raise AttributeError('Topics have already been assigned') 

    def _getSingleTopic(self, i):
        res = self.model[self.news.corpus[i]]
        res.sort(key=lambda x: -x[1])
        return res[0][0]

    def getLogPerplexity(self):
        return self.model.log_perplexity()

    def getCoherence(self, kind='c_v'):
        if kind == 'c_v':
            coherence_model_lda = CoherenceModel(model=self.model, texts=self.news.doc_list, dictionary=self.news.words, coherence='c_v')
        elif kind == 'u_mass':
            coherence_model_lda = CoherenceModel(model=self.model, corpus=self.news.corpus, coherence='u_mass')
        else:
            raise ValueError("Only c_v and u_mass are currently supported")
        return coherence_model_lda.get_coherence()



