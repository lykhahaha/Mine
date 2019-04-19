# USAGE
# python test_prediction.py --image image.jpg
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
import os

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image/ directory')
args = vars(ap.parse_args())

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

    return age_preds, gender_preds

# initialize dlib's face detector (HOG-based), then create facial landmark predictor and face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

# initialize list of image paths
image_paths = [args['image']]

# if input path is directory
if os.path.isdir(args['image']):
    image_paths = sorted(list(paths.list_images(args['image'])))

# initialize image preprocessors
sp, cp, iap = SimplePreprocessor(256, 256, inter=cv2.INTER_CUBIC), CropPreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE, horiz=False), ImageToArrayPreprocessor()

# loop over image paths
for image_path in image_paths:
    # load image fron disk, resize it and convert it to grayscale
    print(f'[INFO] processing {image_path}')
    image = cv2.imread(image_path)
    image = imutils.resize(image, width=1024)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in grayscale image
    rects = detector(gray, 1)

    # loop over face detections
    for rect in rects:
        # determine facial landmarks for face region, then align face
        shape = predictor(gray, rect)
        face = fa.align(image, gray, rect)

        # draw bounding box around face
        clone = image.copy()
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

            # visualize age and gender predictions
            age_canvas = AgeGenderHelper.visualize_age(age_preds, age_le)
            gender_canvas = AgeGenderHelper.visualize_gender(gender_preds, gender_le)
            
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

            # visualize age and gender predictions
            age_canvas = AgeGenderHelper.visualize_age(age_preds, age_le)
            gender_canvas = AgeGenderHelper.visualize_gender(gender_preds, gender_le)

        # show output iamge
        cv2.imshow('Input', clone)
        cv2.imshow('Face', face)
        cv2.imshow('Age Probabilities', age_canvas)
        cv2.imshow('Gender Probabilities', gender_canvas)
        cv2.waitKey(0)