[![Build Status](https://travis-ci.com/yeouchiou/TagThis.svg?branch=main)](https://travis-ci.com/yeouchiou/TagThis)
# TagThis!

TagThis! is an automated news tagging system app that saves editors time by automatically creating topics and classifying articles so that editors can create relevant tags and quickly and easily curate news content. It leverages Latent Dirichlet Allocation [[Blei+03]](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) to create latent topics, then uses a support vector classifier to infer topics from new articles. This app was created during my time at Insight Data Science as an Artificial Intelligence Fellow in 20C.

## Setup

To simply run the app do the following:

1. git clone this repository and create an environment 
```
conda create -n tagthis python=3.6
conda activate tagthis
cd tagthis
```

2. Build Dockerfile
```
docker image build -t tagthis:app .
```

3. Run Dockerfile
```
docker container run -p 8501:8501 -d tagthis:app
```

To end the session, run `docker ps` and find the container ID and then run `docker kill <container_ID>`.

## Advanced Setup/Usage

To run the app on new data, you need to install the package. You may also import your own classifiers.
1. git clone this repository and create an environment 
```
conda create -n tagthis python=3.6
conda activate tagthis
cd tagthis
```

2. Install package
```
python setup.py install
```

3. Example usage:
```
from TagThis import TopicModel, Articles
news = Articles('data/nytimes_news_articles.txt')
lda = TopicModel(news, 10)
```

From there, you can use several `TopicModel` methods such as `generateWordCloud`, `saveAllWordClouds`, `printTopics`, `getLogPerplexity`, and `getCoherence`.

*Note*: If running this on a different dataset, you need to use the `droprows` kwarg in `Articles` and specify in a list the indices of the rows to drop. If there are none, simply pass `droprows=[]`. (There were empty articles that were preprocessed that I had to remove)

## How it works

The streamlit app has two modes. In the "tagger" mode you can paste a full text article and it will assign one of the latent topics from the LDA algorithm. You can then select relevant words as tags. In the "explore tags" option, you can explore all the different topics and tags. 

![image](images/tagger.png)
![image](images/exploretags.png)

## Topic Modeling via Latent Dirichlet Allocation

Latent Dirichlet Allocation is a generative statistical model which creates latent topics from a corpus of docments. LDA assumes that each each document is a combination of different topics. So say document1 could be 50% topic1, 30% topic2, and 20% topic3. The topics themselves are abstract and can be represented by different words. For example, there might be a CAT_related topic which has probabilities to generate words such as milk, meow, or kitten. Given a preprocessed corpus of documents in bag of words representation, LDA outputs word probabilities for each topic as well as assigns the documents to mixtures of topics. 

![image](images/lda.png)

I trained LDA on a Kaggle New York Times article dataset which contained around 9000 full-text articles from April to June in 2016. Articles had an average of about 800 words per article. I used average Jaccard similarity between topics to choose the number of topics. From the elbow method, 10 topics were chosen for this dataset.

![image](images/jaccard.png)

## Classifier

Assuming that these topics are ground truth labels, I then trained a support vector machine (SVM) classifier with a Tf-Idf feature representation for the articles. SVMs allow for lightweight and fast inference due to only requiring the trained support vectors. The SVM achieves a micro-averaged precision of 88.6%!
