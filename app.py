import streamlit as st
import pandas as pd
import numpy as np
import gensim.corpora as corpora
from gensim.models import TfidfModel, LdaModel
from gensim import similarities
from TagThis.preprocess import *
from wordcloud import WordCloud
import pickle
import joblib

st.title('TagThis!')
df = preprocessToDF()
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the app mode", ['Tagger', 'Explore Tags'])

def identity(x): return x

def loadCorpus():
    with open('data/processedtext.pkl', 'rb') as f:
        doc_list, words, corpus = pickle.load(f)
    return doc_list, words, corpus

def loadModels():
    model = LdaModel.load('models/ldamodels_bow_10.lda')
    tfidf = joblib.load('models/tfidfvectorizer.joblib')
    svm = joblib.load('models/svm.joblib')
    return model, tfidf, svm

doc_list, words, corpus = loadCorpus()
model, tfidf, svm = loadModels()

def loadTagger():
    "### Welcome to TagThis!, an app that automatically tags your news article for content curation!"
    "#### On the left, you can choose whether you want to tag an article or explore what the tag topics look like."
    "#### Paste a full text article to get tagging!"
    user_input = st.text_area("", height=100)

    if user_input:
        processedDoc = preprocessSingleInput(user_input)
        doctfidf = tfidf.transform(np.array(processedDoc).reshape(1,-1))
        tag = svm.predict(doctfidf)[0]
        'This article belongs to Topic #' + str(tag) + '!'
        st.image('images/LDAtopic'+str(tag)+'.jpg')
        

def exploreTags():
    st.markdown("### Explore what words are represesnted by the latent topics!")
    for t in range(model.num_topics):
        st.image('images/LDAtopic'+str(t)+'.jpg')

if app_mode == 'Tagger':
    loadTagger()
elif app_mode == 'Explore Tags':
    exploreTags()
