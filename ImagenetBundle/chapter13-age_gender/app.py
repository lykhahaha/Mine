from flask import Flask, render_template, request
import os
import shutil
import cv2
from config import age_gender_deploy as deploy
from config import age_gender_config as config
from pyimagesearch.preprocessing import SimplePreprocessor, MeanPreprocessor, CropPreprocessor, ImageToArrayPreprocessor
from pyimagesearch.utils import AgeGenderHelper
from pyimagesearch.callbacks import OneOffAccuracy
from keras.models import load_model
from imutils.face_utils import FaceAligner
from imutils import face_utils, paths
import numpy as np
import argparse
import imutils
import pickle
import json
import dlib
import urllib
import shutil
from flask_ngrok import run_with_ngrok
from keras import backend as K
import gc

app = Flask(__name__)
# run_with_ngrok(app)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict(face, sp, age_mp, gender_mp, cp, iap, age_path, gender_path, age_le, gender_le):
    # load model from disk
    gender_model = load_model(gender_path)

    agh = AgeGenderHelper(config, deploy)
    one_off_mappings = agh.build_oneoff_mappings(age_le)
    one_off = OneOffAccuracy(one_off_mappings)
    custom_objects={'one_off_accuracy': one_off.one_off_accuracy}
    age_model = load_model(age_path, custom_objects=custom_objects)

    # resize and crop image
    age_crops = cp.preprocess(age_mp.preprocess(sp.preprocess(face)))
    age_crops = np.array([iap.preprocess(c) for c in age_crops])

    gender_crops = cp.preprocess(gender_mp.preprocess(sp.preprocess(face)))
    gender_crops = np.array([iap.preprocess(c) for c in gender_crops])

    # predict on age and gender based on extracted crops
    age_preds = age_model.predict(age_crops).mean(axis=0)
    gender_preds = gender_model.predict(gender_crops).mean(axis=0)

    del gender_model, age_model

    return age_preds, gender_preds

@app.route('/')
def hello_world():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    
    image_path = os.path.sep.join([UPLOAD_FOLDER, file.filename])
    file.save(image_path)
    # image_url = uploader.upload(image_path)
    # image = AgeGenderHelper.url_to_image(image_url['url'])

    # initialize dlib's face detector (HOG-based), then create facial landmark predictor and face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
    fa = FaceAligner(predictor)

    # initialize image preprocessors
    sp, cp, iap = SimplePreprocessor(256, 256, inter=cv2.INTER_CUBIC), CropPreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE, horiz=False), ImageToArrayPreprocessor()

    # loop over image paths
    # load image fron disk, resize it and convert it to grayscale
    print(f'[INFO] processing {file.filename}')
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=1024)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clone = image.copy()

    # detect faces in grayscale image
    rects = detector(gray, 1)

    # loop over face detections
    for rect in rects:
        # determine facial landmarks for face region, then align face
        shape = predictor(gray, rect)
        face = fa.align(image, gray, rect)

        # draw bounding box around face
        x, y, w, h = face_utils.rect_to_bb(rect)
        cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if config.DATASET == 'IOG':
            # load Label Encoder and mean files
            print('[INFO] loading label encoders and mean files...')
            age_le = pickle.loads(open(deploy.AGE_LABEL_ENCODER, 'rb').read())
            gender_le = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, 'rb').read())
            age_means = json.loads(open(deploy.AGE_MEAN).read())
            gender_means = json.loads(open(deploy.GENDER_MEAN).read())

            # initialize image preprocessors
            age_mp = MeanPreprocessor(age_means['R'], age_means['G'], age_means['B'])
            gender_mp = MeanPreprocessor(gender_means['R'], gender_means['G'], gender_means['B'])

            age_preds, gender_preds = predict(face, sp, age_mp, gender_mp, cp, iap, deploy.AGE_NETWORK_PATH, deploy.GENDER_NETWORK_PATH, age_le, gender_le)
            
        elif config.DATASET == 'ADIENCE':
            # age_preds_cross, gender_preds_cross = [], []

            i = 0
            # load Label Encoder and mean files
            print(f'[INFO] loading label encoders and mean files for cross validation {i}...')
            age_le = pickle.loads(open(deploy.AGE_LABEL_ENCODERS[i], 'rb').read())
            gender_le = pickle.loads(open(deploy.GENDER_LABEL_ENCODERS[i], 'rb').read())
            age_means = json.loads(open(deploy.AGE_MEANS[i]).read())
            gender_means = json.loads(open(deploy.GENDER_MEANS[i]).read())

            # initialize image preprocessors
            age_mp = MeanPreprocessor(age_means['R'], age_means['G'], age_means['B'])
            gender_mp = MeanPreprocessor(gender_means['R'], gender_means['G'], gender_means['B'])

            age_preds, gender_preds = predict(face, sp, age_mp, gender_mp, cp, iap, deploy.AGE_NETWORK_PATHS[i], deploy.GENDER_NETWORK_PATHS[i], age_le, gender_le)
            # age_preds_cross.append(age_pred)
            # gender_preds_cross.append(gender_pred)

            # age_preds, gender_preds = np.mean(age_preds_cross, axis = 0), np.mean(gender_preds_cross, axis = 0)

        clone = AgeGenderHelper.visualize_video(age_preds, gender_preds, age_le, gender_le, clone, (x, y))

    # path = image_path.split('.')
    # pred_path = '.'.join([f'{path[0]}_predict', path[1]])
    # pred_filename = pred_path.split(os.path.sep)[-1]
    pred_path = '.'.join([f"{image_path.split('.')[0]}_1", 'jpg'])
    cv2.imwrite(pred_path, clone)
    # image_url = uploader.upload(pred_path)
    gc.collect()
    K.clear_session()

    return render_template('index.html', filename=pred_path.split(os.path.sep)[-1])

if __name__ == '__main__':
    app.run()