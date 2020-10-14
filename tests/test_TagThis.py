import numpy as np
import numpy.testing as npt
import TagThis

def test_Articles_smoke():
    # Smoke test
    obj = TagThis.Articles('data/test.txt', droprows=[])

def test_TagThis_smoke():
    # Smoke test
    news = TagThis.Articles('data/test.txt', droprows=[])
    obj = TagThis.TopicModel(1, news)
