import time
import io
import os
from os import walk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from PIL import Image
from torch.autograd import Variable
import redis

import flask
from flask import request, redirect, url_for

class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

app = flask.Flask(__name__)
model = None
email = None
file_list = []
answers = []
lng = 'ru'
trans = {
    'ru':{
        'right_answer':'верно',
        'wrong_answer':'неверно',
        'answer':{
            0:'признаки отсутствуют',
            1:'признаки присутствуют',
            3:'нет ответа'
        }
    }
}
r = redis.StrictRedis(host= 'localhost', decode_responses=True)

def load_model():
    global model

    model = DenseNet121(2).cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('TB.model'))

def generate_file_list():
    global file_list
    global answers

    #for (dirpath, dirnames, filenames) in walk('static/'):
    #    file_list.extend(filenames)
    #    break
    
    file_list = pd.read_csv('file_list.csv').values
    file_list = np.random.permutation(file_list)
    answers = [3] * len(file_list)

@app.route("/check/")
def check():
    return (
        '<form enctype="multipart/form-data" action="predict" method="post"><input type="file" name="image" /><input type="submit"  /></form>')

@app.route("/uvsai/", methods= ['POST', 'GET'])
def uvsai():
    
    return flask.render_template('uvsai.html')

@app.route("/uvsai/<int:img_id>", methods= ['POST', 'GET'])
def uvsai_id(img_id):
    #global trans
    #global lng
    global email
    global file_list
    global answers
    
    if request.method == 'POST':
        if request.form.get('email') != None:
            email = request.form.get('email')
            
            # новая сессия
            if r.hget(email, 'session') == None: 
                
                # создаем запись о сессии
                r.rpush('players', email)
                r.hset(email, 'session', time.time())
                
                # генерируем и сохраняем список изображений и ответов
                generate_file_list()
                r.rpush(email+':image_path', *file_list[:,0])
                r.rpush(email+':image_gt', *file_list[:,1])
                r.rpush(email+':image_pred', *file_list[:,2])
                r.rpush(email+':answers', *answers)
                
                # генерируем счетчики для пользователя и ИИ
                r.hset(email, 'u_atotal', 0)
                r.hset(email, 'ai_atotal', 0)
                r.hset(email, 'u_ca', 0)
                r.hset(email, 'ai_ca', 0)
            
            # продолжение сессии
            else: 
                generate_file_list()
                file_list[:,0] = r.lrange(email+':image_path', 0, -1)
                file_list[:,1] = r.lrange(email+':image_gt', 0, -1)
                file_list[:,2] = r.lrange(email+':image_pred', 0, -1)
                answers = r.lrange(email+':answers', 0, -1)
    
    #print(email)
    if (email == None) | (img_id not in set(range(len(file_list)))):
        return(redirect(url_for('uvsai'), code=302))
        
    image_path = file_list[img_id][0]
    image_gt = file_list[img_id][1]
    image_pred = file_list[img_id][2]
    laquo = img_id - 1 if img_id > 0 else img_id
    raquo = img_id + 1 if img_id < (len(file_list) - 1) else img_id
    
    vardict = {
        'img_id':img_id,
        'laquo':laquo,
        'raquo':raquo,
        'image_path':image_path,
        'image_gt':image_gt,
        'image_pred':image_pred,
    }
    
    if request.method == 'POST': 
        if request.form.get('answer') in ('0','1'):
            if (int(answers[img_id]) == 3):
                r.hincrby(email, 'u_atotal', 1)
                r.hincrby(email, 'ai_atotal', 1)
                r.hincrby(email, 'u_ca', int(image_gt == int(request.form.get('answer'))))
                r.hincrby(email, 'ai_ca', int(image_gt == image_pred))
                answers[img_id] = int(request.form.get('answer'))
                r.lset(email+':answers', img_id, int(request.form.get('answer')))
            else:
                pass
    #vardict['image_answer_u'] = trans[lng]['answer'][int(answers[img_id])]
    #vardict['image_answer_ai'] = trans[lng]['answer'][int(image_pred)]
    vardict['image_answer_u'] = trans[lng]['right_answer'] if int(answers[img_id]) == int(image_gt) else trans[lng]['wrong_answer']
    vardict['image_answer_ai'] = trans[lng]['right_answer'] if int(image_pred) == int(image_gt) else trans[lng]['wrong_answer']
    vardict['image_answer_gt'] = trans[lng]['answer'][int(image_gt)]
    vardict['image_answer'] = int(answers[img_id])
    vardict['total_num_of_imgs'] = len(file_list)
    #print(vardict)
    
    result = {
        'u':{'player':'Вы'},
        'ai':{'player':'ИИ'}
    }
    for key in result.keys():
        val = r.hgetall(email)
        result[key]['ca'] = int(val[key+'_ca'])
        result[key]['atotal'] = int(val[key+'_atotal'])
        result[key]['acc'] = float(int(val[key+'_ca']) / int(val[key+'_atotal']) if int(val[key+'_atotal']) > 0 else 0) * 100
        
        # !!!КОСТЫЛЬ!!!
        result[key]['atime'] = int(0)
    
    return flask.render_template('uvsai_id.html', result= result, vardict= vardict)

@app.route("/predict", methods= ['POST', 'GET'])
def predict():
    data = {'success': False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image)).convert('RGB')

            model.eval()

            normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
            TBtransforms1 = transforms.Compose([
                transforms.Resize((256, 256)),
                # transforms.ToTensor()
                transforms.FiveCrop(224),
                transforms.Lambda
                (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda
                (lambda crops: torch.stack([normalize(crop) for crop in crops]))
            ])

            TBtransforms2 = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

            inputs = TBtransforms2(image)

            inputs.unsqueeze_(0);
            inputs = Variable(inputs).cuda()
            outputs = model(inputs)

            preds = outputs.mean(0).cpu().data.numpy()

            data['tuberculosis'] = int(preds[1] > preds[0])
            data['probaility'] = float(preds[1])

            data['success'] = True

    return flask.jsonify(data)

if __name__ == "__main__":
    #print('Loading CheXNet model...')
    #load_model()
    #print('Model loaded!')
    #generate_file_list()
    #print(len(file_list))
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port)
    #app.run(host='0.0.0.0', port=port, threaded=True)