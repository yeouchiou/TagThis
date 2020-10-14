import numpy as np
import numpy.testing as npt
import TagThis

def test_Articles_smoke():
    # Smoke test
    obj = Articles('data/test.txt')

def test_TagThis_smoke():
    # Smoke test
    news = Articles('data/test.txt')
    obj = TagThis.TopicModel(1, news)
