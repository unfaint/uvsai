#!/usr/bin/python3

import time
import io
import os
import numpy as np
import pandas as pd
from PIL import Image
import redis

import flask
from flask import request, redirect, url_for, session
from werkzeug.utils import secure_filename


MODEL_PATH = 'TB.model'
FILE_LIST_PATH = 'file_list.csv'

FLASK_SECRET = 'uvsai'


def log_msg(msg, level=1):
    if level == 1:
        print(msg)
    if level == 0:
        pass
    else:
        pass


def create_file_list_and_answers(file_list_path):
    file_list = pd.read_csv(file_list_path).values
    file_list = np.random.permutation(file_list)
    answers = [3] * len(file_list)
    return file_list, answers


class UvsaiModel:
    def __init__(self):
        self.file_list, self.answers = create_file_list_and_answers(FILE_LIST_PATH)
        self.email = None
        self.r = redis.StrictRedis(host='localhost', decode_responses=True)


class UvsaiPresenter:
    def __init__(self, model):
        self.model = model


model = UvsaiModel()
presenter = UvsaiPresenter(model)
app = flask.Flask(__name__)


@app.route("/uvsai/", methods=['POST', 'GET'])
def uvsai():
    return flask.render_template('uvsai.html')


if __name__ == "__main__":
    app.secret_key = FLASK_SECRET
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

