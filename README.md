# Optimizing your web conference background with Room Rater

#### Data Science Nanodegree Capstone Project

## Project Definition

### Overview

This project predicts the quality of people's web conference backgrounds using Natural Language Processing and Machine Learning pipelines.

The models are trained and tested on tweets collected from the Room Rater (@ratemyskyperoom) Twitter account. This account posts photos of people's web conference backgrounds, critiquing the background aesthetics and assigning them a score of 0-10 out of 10.

Quick examples of Room Rater's tweets and rating style:
[image 8 and 9]

Natural Language Processing is used to tokenize the tweet text to identify key vocabularly used in the background evaluation criteria. The tokenized text is also fed into the machine learning model, where several classifiers are tested in their ability to predict ratings.

#### Navigation

##### README Contents

- [Optimizing your web conference background with Room Rater](#optimizing-your-web-conference-background-with-room-rater)
      - [Data Science Nanodegree Capstone Project](#data-science-nanodegree-capstone-project)
  - [Project Definition](#project-definition)
    - [Overview](#overview)
      - [Navigation](#navigation)
        - [README Contents](#readme-contents)
        - [Files](#files)
    - [Problem Statement](#problem-statement)
    - [Metrics](#metrics)
    - [Quick start](#quick-start)
      - [Access tweets with the Twitter API](#access-tweets-with-the-twitter-api)
      - [Requirements](#requirements)
  - [Methodology](#methodology)
  - [Results](#results)
  - [Acknowledgements](#acknowledgements)

##### Files

- RoomRater.ipynb - contains code for data wrangling, NLP, models, and evaluation as well as analysis of findings along the way
- Data sources:
    - roomratertweets.csv (tweets collected through the Twitter API)
    - roomratertweets2.csv (additoinal tweets collected)
- Blog post:

### Problem Statement

At the start of the COVID pandemic, professionals found themselves suddenly launched into a work-from-home situation, requireing that they attend meetings from their kitchens, living rooms, bedrooms, and if they're lucky, home offices.

Many people were not used to presenting themselves in this context, and are even less aware of how their bacgrounds were a part of the impression they gave. 

Enter Room Rater. This Twitter account begain posting photos of various people appearning on air, particularly reporters. They began scoring people's backgrounds, applauding them for a good use of plants and books in the background, or critiquing their lighighting situation.

One can scroll through to get an idea about what might make a good background, but what would the data say if we look at their scores systematically.

### Metrics

**Feature**: Tokenized tweet text

**Outcome variable**: Rating, a multiclass variable on a 1-10 scale

To evaluate the classification models, the following **evaluation metrics** will be used:
- Accuracy - portion of labels accurately predicted
- Precision - the portion of predictions of a specific class that are correctly predicted (ex. how many predicted to be 9 were actually 9)
- Recall - the portion of a specific class a correctly predicted (ex. how many actual 9's were predicted to be 9)
- F1 score - the harmonic mean of precision and recall
- ROC AUC - the area under the ROC curve (true positive rate vs false positive rate), with .5 signifying the model performing on par with random classification.

Because the rating is ordinal, the above metrics don't account for the degree of misclassification. They won't recognize that misclassifying a 10 as a 9 is perferable to misclassifying a 10 as a 2. Therefore we'll also examine the following for evaluation:
- Average absolute value of the difference between the actual and predicted ratings

### Quick start

#### Access tweets with the Twitter API

Resources to get you started with the Twitter API and a helpful python package for getting tweets:
https://developer.twitter.com/en/docs/twitter-api/tweets/lookup/introduction
https://docs.tweepy.org/en/latest/
https://docs.tweepy.org/en/v3.10.0/cursor_tutorial.html

#### Requirements

Clone the repository and install the requirements using the script below. The project uses Python 3.8.8.

`pip install -r requirements.txt`

## Methodology

Several **Natural language processing** techiques were used to identify relevant key words:
- Removal of punctuation, URLs, and other non-text characters, as well as normalization of case
- Removal of English stop words (the, a, an)
- Word tokenization to break sentences into word tokens for analysis
- Lemmatization so that words like plants and plant can be grouped into a single token

[image]
We can tell RoomRater cares most about art in the background, followed by plants, books, and a sense of depth. After that, lighting, pillows, and flowers factor in.

We can also see how RoomRater focuses on different keywords for backgrounds of different quality. Low- to mid-rated backgrounds need work on camera angle and keeping their cords out of sight. Backgrounds in the 7-9 range have the basics down and can focus on adding elements like plants and art to enhance the decor.
[image]

**Five classifiers were evaluated:**
- Random Forest Classifier (fits multipe decision tree classifiers on different sub-samples to minimize over-fitting)
- Balanced Random Forest Classifier (balances by employing under-sampling to the random forest classifier)
- Gradient Boosting Classifier (runs multiple Decision Tree classifiers to minimize the loss function)
- Easy Ensemble Classifier (using the AdaBoost Classifier as a base estimator, employs random under-sampling on the bootstrap samples)
- Ordinal Logistic Regresstion (a classifier that takes into account that the order of the ratings are meaningful)

GridSearchCV was implemented to evaluate several combinations of parameters for the classifiers.

## Results

The Random Forest Classifier was the only classifier that had an ROC AUC score above .6. With .5 signifying performance equivalent to random assignment, none of these models performed great.

In general, the balanced classifiers did not perform as well as the originals. Here's an example of the Random Forest and the Balanced Random Forest predictions in comparison with the actual ratings:
[image_rf]

If the real world population is similar to the sample that RoomRater has collected, with many excellent, 10/10-worthy backgrounds, the Random Forest Classifier shows the most potential. However, if people's web backgrounds in the real world are less likely to score a 10, the balanced classifiers may be worth considering.

See Jupyter Notebook for full evaluation of each model.

## Acknowledgements

This projec was completed as a part of the Data Science Nanodegree with www.udacity.com.

