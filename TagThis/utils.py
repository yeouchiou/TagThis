def jaccard_similarity(setA, setB):
    intersection = set(setA).intersection(set(setB))
    union = set(setA).union(set(setB))
    return float(len(intersection))/float(len(union))


def makeWordcloudImages(model, path):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    for i in range(model.num_topics):
        fig, ax = plt.subplots()
        ax.imshow(WordCloud().fit_words(dict(model.show_topic(i, 200))))
        ax.axis("off")
        ax.set_title("Topic #" + str(i))
        plt.savefig(path + 'LDAtopic' + str(i) + '.jpg')
