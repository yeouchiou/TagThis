import numpy as np
import numpy.testing as npt

import TagThis

def test_TagThis_smoke():
    #Smoke test
    doc_list, words, corpus = TagThis.preprocess.createCorpus()
    obj = TagThis.TopicModel(10, doc_list, words, corpus)
