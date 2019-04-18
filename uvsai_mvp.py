#!/usr/bin/python3

import os
import numpy as np
import pandas as pd
import redis
from time import time

import flask
from flask import request, redirect, url_for, session


MODEL_PATH = 'TB.model'
FILE_LIST_PATH = 'file_list.csv'

FLASK_SECRET = 'uvsai'


def log_msg(msg, level=1):
    if level > 0:
        print(msg)
    else:
        pass


def create_file_list_and_answers(file_list_path):
    file_list = pd.read_csv(file_list_path)
    answers = np.array([3] * len(file_list))
    return file_list, answers


class UvsaiModel:
    def __init__(self):
        self.file_list, self.answers = create_file_list_and_answers(FILE_LIST_PATH)
        self.file_ixs = None
        self.gt = None
        self.pred = None
        self.email = None
        self.r = redis.StrictRedis(host='localhost', decode_responses=True)

    def check_email(self):
        return self.r.hget(self.email, 'session') is not None

    def set_gt_and_pred(self):
        self.gt = self.file_list.loc[self.file_ixs, 'labels']
        self.pred = self.file_list.loc[self.file_ixs, 'pred']

    def r_get_saved_data(self):
        self.file_ixs = np.array([int(i) for i in self.r.lrange(self.email + ':file_ixs', 0, -1)])
        self.set_gt_and_pred()

        self.answers = np.array([int(i) for i in self.r.lrange(self.email + ':answers', 0, -1)])

    def r_create_session(self):
        self.file_ixs = np.random.permutation(self.file_list.index)
        self.set_gt_and_pred()

        self.r.rpush('players', self.email)
        self.r.hset(self.email, 'session', time())
        self.r.rpush(self.email + ':file_ixs', *self.file_ixs)
        self.r.rpush(self.email + ':answers', *self.answers)

        for i in ['u_atotal', 'ai_atotal', 'u_ca', 'ai_ca']:
            self.r.hset(self.email, i, 0)

    def r_add_answer(self, img_id, answer):
        self.r.hincrby(self.email, 'u_atotal', 1)
        self.r.hincrby(self.email, 'ai_atotal', 1)

        self.r.hincrby(self.email, 'u_ca', int(self.gt[img_id] == answer))
        self.r.hincrby(self.email, 'ai_ca', int(self.gt[img_id] == self.pred[img_id]))

        self.answers[img_id] = answer
        self.r.lset(self.email + ':answers', img_id, answer)

    def metrics(self, ai=False):
        y_true = self.file_list.loc[self.file_ixs, 'labels']
        if ai:
            y_pred = self.file_list.loc[self.file_ixs, 'pred']
            y_pred[self.answers == 3] = 3
        else:
            y_pred = self.answers

        TP = sum((y_true == 1) & (y_pred == 1))
        TN = sum((y_true == 0) & (y_pred == 0))

        FP = sum((y_true == 0) & (y_pred == 1))
        FN = sum((y_true == 1) & (y_pred == 0))

        total = TP + TN + FP + FN
        correct = TP + TN

        accuracy = float(correct / total) * 100 if total > 0 else 0.
        precision = float(TP / (TP + FP)) if (TP + FP) > 0 else .0
        recall = float(TP / (TP + FN)) if (TP + FN) > 0 else .0
        f1 = float(2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else .0

        return {'ca': correct, 'atotal': total, 'acc': accuracy, 'pre': precision, 'rec': recall, 'f1': f1}


class UvsaiPresenter:
    def __init__(self, model):
        self.model = model
        self.answer_text = {
            0: 'признаки отсутствуют',
            1: 'признаки присутствуют',
            3: 'нет ответа'
        }

    def enter_session(self, email=None):  # TODO validate email
        if email is None:
            if bool(session['email']):
                self.model.email = session['email']
            else:
                return False
        else:
            session['email'] = self.model.email = email

        continue_session = self.model.check_email()

        if continue_session:
            self.model.r_get_saved_data()
        else:
            self.model.r_create_session()

        return True

    def check_img_id(self, img_id):
        return (0 <= img_id) & (img_id < len(self.model.file_ixs))

    def add_answer(self, img_id, answer):
        self.model.r_add_answer(img_id, int(answer))

    def get_vardict(self, img_id):
        ix = self.model.file_ixs[img_id]

        image_gt = self.model.file_list.loc[ix, 'labels']
        image_pred = self.model.file_list.loc[ix, 'pred']

        vardict = dict()
        vardict.update({
            'image_path': self.model.file_list.loc[ix, 'names'],
            'image_gt': image_gt,
            'image_pred': image_pred,
            'laquo': img_id - 1 if img_id > 0 else len(self.model.file_ixs) - 1,
            'raquo': img_id + 1 if img_id < (len(self.model.file_ixs) - 1) else 0,

            'image_answer_u': 'верно' if int(self.model.answers[img_id]) == int(image_gt) else 'неверно',
            'image_answer_ai': 'верно' if int(image_pred) == int(image_gt) else 'неверно',
            'image_answer_gt': self.answer_text[int(image_gt)],
            'image_answer': int(self.model.answers[img_id]),
            'total_num_of_imgs': len(self.model.file_ixs),
        })

        return vardict

    def get_results(self):
        results = dict()
        results.update({
            'u': {},
            'ai': {}
        })

        results['u'] = self.model.metrics(ai=False)
        results['ai'] = self.model.metrics(ai=True)

        results['u']['player'] = 'Вы'
        results['ai']['player'] = 'ИИ'

        return results


model_ = UvsaiModel()
presenter = UvsaiPresenter(model_)
app = flask.Flask(__name__)


@app.route("/uvsai/", methods=['POST', 'GET'])
def uvsai():
    return flask.render_template('uvsai.html')


@app.route("/uvsai/<int:img_id>", methods=['POST', 'GET'])
def uvsai_id(img_id):
    global presenter

    email = None

    if request.method == 'POST':
        if bool(request.form.get('email')):
            email = request.form.get('email')

    if presenter.enter_session(email):
        if presenter.check_img_id(img_id):

            if request.method == 'POST':
                if request.form.get('answer') in ('0', '1'):
                    presenter.add_answer(img_id, request.form.get('answer'))

            vardict = presenter.get_vardict(img_id)
            results = presenter.get_results()
            return flask.render_template('uvsai_id.html', result=results, vardict=vardict)
    else:
        return redirect(url_for('uvsai'), code=302)


if __name__ == "__main__":
    app.secret_key = FLASK_SECRET
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
