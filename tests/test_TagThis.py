import numpy as np
import numpy.testing as npt
import TagThis

def test_TagThis_smoke():
    #Smoke test
    from gensim.test.utils import common_texts
    from gensim.corpora.dictionary import Dictionary
    common_dictionary = Dictionary(common_texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in common.texts]

    obj = TagThis.TopicModel(1, common_texts, common_dictionary, common_corpus)
