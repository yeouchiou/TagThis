[![Build Status](https://travis-ci.com/yeouchiou/TagThis.svg?branch=main)](https://travis-ci.com/yeouchiou/TagThis)
# TagThis!

TagThis! is an automated news tagging system app that saves editors time by automatically creating topics and classifying articles so that editors can create relevant tags and quickly and easily curate news content. It leverages Latent Dirichlet Allocation to create latent topics, then uses a support vector classifier to infer topics from new articles. 

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
## Advanced Setup
---
To run the app on new data, you need to install the package. You may also import your own classifiers.
1. git clone this repository and create an environment 
```
conda create -n tagthis python=3.6
conda activate tagthis
cd tagthis
```

2. Install package
`
python setup.py install
`

## How it works

The app has two modes. OIn the "tagger" mode you can paste a full text article and it will assign one of the latent topics from the LDA algorithm. You can then select relevant words as tags. In the "explore tags" option, you can explore all the different topics and tags. 

![image](images/tagger.png)
![image](images/exploretags.png)
