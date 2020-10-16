import json
from spam_detection import model_tools, models
import flask
from flask import Flask, request, Response
import jsonpickle
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = load_model(r'models/spam_model')

app = flask.Flask(__name__)
app.config["DEBUG"] = True


class_mapping = {0: 'ham',
                 1: 'spam'}


@app.route('/detect_spam', methods=['POST'])
def test():
    r = request
    print(r.data)

    data_string = r.data.decode("utf-8")
    data_string = [data_string]

    message_preprocessed = model_tools.preprocess_input(data_string)

    y_pred = model.predict(message_preprocessed)
    result = np.round(y_pred[0])
    result = class_mapping[result[0]]

    response_json = json.dumps(result, indent=4)

    return Response(response=response_json, status=200, mimetype="application/json")


@app.route('/', methods=['GET'])
def home():
    return "Spam detection system - to use, send POST with message on /detect_spam"


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0')
