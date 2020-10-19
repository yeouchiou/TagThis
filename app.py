import streamlit as st
import numpy as np
import joblib
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models import LdaModel


def preprocessSingleInput(data):
    # assumes data is a string of sentences
    if len(data) < 10:
        raise ValueError('Please post an article with more than a 10 words')
    # remove city information
    if data.split()[1] == '—':
        data = ' '.join(data.split()[2:])
    # remove punctuation
    data = re.sub('[,\.!?]', '', data)
    # lowercase everything
    data = data.lower()
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


def identity(x): return x


def loadModels():
    model = LdaModel.load('models/ldamodels_bow_10.lda')
    tfidf = joblib.load('models/tfidfvectorizer.joblib')
    svm = joblib.load('models/svm.joblib')
    return model, tfidf, svm


def loadTagger():
    "### Welcome to TagThis!, an app that automatically tags your news article for content curation!"
    "#### On the left, you can choose whether you want to tag an article or explore what the tag topics look like."
    "#### Paste a full text article to get tagging!"
    user_input = st.text_area("", height=100)

    if user_input:
        processedDoc = preprocessSingleInput(user_input)
        doctfidf = tfidf.transform(np.array(processedDoc).reshape(1, -1))
        tag = svm.predict(doctfidf)[0]
        'This article belongs to Topic #' + str(tag) + '!'
        st.image('images/LDAtopic'+str(tag)+'.jpg')


def exploreTags():
    st.markdown("### Explore what words are represesnted by the latent topics!")
    for t in range(model.num_topics):
        st.image('images/LDAtopic'+str(t)+'.jpg')


st.title('TagThis!')
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the app mode", ['Tagger', 'Explore Tags'])
model, tfidf, svm = loadModels()

if app_mode == 'Tagger':
    loadTagger()
elif app_mode == 'Explore Tags':
    exploreTags()
