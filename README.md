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

To launch prediction server, type:
```
python -m server.app
```
This launches a Flask server. If the server is up and running, you may communicate with it like in `rest_client.ipynb`

You can also set up an entire project in Docker container. For that, you first need to build it:
```
docker build -t spam_detection -f docker/development.dockerfile .
```
Then, you can launch a Flask server in Docker container by typing:
```
docker run -p 5000:5000  spam_detection python -m server.app
```
To and visit `http://localhost:5000/` to check if the server is running.

## Methods

This could be interpreted a real-world scenario of applying Recurrent Neural Networks for sequence data. The project contains experiments, analysis, training and serving all in one.

## Stack

In the project, Tensorflow together with high-level abstractions written in Keras were used. As Keras makes it very convenient
to quickly build NLP solutions, some handy out-of-the-box tools were used. However, a standard step-by-step approach for
data preparation & preprocessing were used to show how it's done.

## Coming soon
The whole project will be dockerized soon.

### Endnote
You may check some of my different projects on my [Github Pages Site](https://wprazuch.github.io/)

