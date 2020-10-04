# Spam Detection in E-mails and Instant Messages
![Spam](static/spam.png)


## Introduction

This repository contains a simple workflow for constructing a Deep Learning model dedicated for spam detection. 
During project, some problem analysis and experiments were taken to evaluate, how SMS spam is different from
Mail spam and how it is affecting the spam detection model. Some scripts to generate the model, and necessary
objects for data handling (which are pickled in the repository and loaded on demand) were provided.

Additionally, a Flask app was added to provide endpoints for communication with the model.

## Data

Datasets for this mini-project were taken from [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) and
[Spam Mails Dataset](https://www.kaggle.com/venky73/spam-mails-dataset).

## How to use

Generally, the data, the model together with necessary pickled obkects for preprocessing were uploaded to the repository. In order to
create them manually, you may generate them using:
```
python train.py
```
There are two parameters which can be passed to training script, `MAXLEN` and `save_path`. `MAXLEN` specifies, how many starting words should be taken as an input
for each observation, whereas `save_path` specifies where to save the trained model. By default, these are set as `100`, and `models/spam_model`. You may check the arguments using `help` on the script.


## Methods

This could be interpreted a real-world scenario of applying Recurrent Neural Networks for sequence data. As a comparison,
a 1D Convolutional layers were also used and explained in the notebooks.

## Stack

In the project, Tensorflow together with high-level abstractions written in Keras were used. As Keras makes it very convenient
to quickly build NLP solutions, some handy out-of-the-box tools were used. However, a standard step-by-step approach for
data preparation & preprocessing were used to show how it's done.

### Endnote
You may check some of my different projects on my [Github Pages Site](https://wprazuch.github.io/)

