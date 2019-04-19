# USAGE
# python vis_classification.py --sample-size 20

import cv2

from config import age_gender_deploy as deploy
from config import age_gender_config as config
from pyimagesearch.preprocessing import SimplePreprocessor, MeanPreprocessor, ImageToArrayPreprocessor, CropPreprocessor
from pyimagesearch.utils import AgeGenderHelper
from pyimagesearch.callbacks import OneOffAccuracy
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import argparse
import pickle
import imutils
from imutils import paths
import json
import os
import h5py
from scipy.io import loadmat
import glob
from imutils import face_utils
from imutils.face_utils import FaceAligner
import dlib

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--sample-size', type=int, default=20, help='number of images used for sampling from Adience')
args = vars(ap.parse_args())

def prediction(config, face, sp, age_mp, gender_mp, cp, age_model, gender_model):
    # preprocess image
    age_image = age_mp.preprocess(sp.preprocess(face))
    # age_image = age_mp.preprocess(clone)
    age_crops = cp.preprocess(age_image)
    age_crops = np.array([iap.preprocess(c) for c in age_crops])

    gender_image = gender_mp.preprocess(sp.preprocess(face))
    if config.DATASET == 'IOG':
        gender_image = cp.preprocess(gender_image)
        gender_image = np.array([iap.preprocess(c) for c in gender_image])
    elif 'ADIENCE' in config.DATASET:
        gender_image = np.expand_dims(iap.preprocess(gender_image), axis=0)

    # pass ROIs through their respective models
    age_preds = age_model.predict(age_crops).mean(axis=0)
    if config.DATASET == 'IOG':
        gender_preds = gender_model.predict(gender_image).mean(axis=0)
    elif 'ADIENCE' in config.DATASET:
        gender_preds = gender_model.predict(gender_image)[0]

    return age_preds, gender_preds

if config.DATASET == 'IOG':
    # load age/gender test
    print('[INFO] loading gender/age test set path...')
    mat = loadmat(config.MAT_TEST_PATH)
    image_paths = [p[0].split('\\')[-1] for p in mat['tecoll'][0][0]['name'][0]]
    rows = np.random.randint(low=0, high=len(image_paths), size=args['sample_size'])

    # load Label Encoder and mean files
    print('[INFO] loading label encoders and mean files...')
    age_le = pickle.loads(open(deploy.AGE_LABEL_ENCODER, 'rb').read())
    gender_le = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, 'rb').read())
    age_mean = json.loads(open(deploy.AGE_MEAN).read())
    gender_mean = json.loads(open(deploy.GENDER_MEAN).read())

    # load model from disk
    age_path = deploy.AGE_NETWORK_PATH
    gender_path = deploy.GENDER_NETWORK_PATH
    gender_model = load_model(gender_path)

    agh = AgeGenderHelper(config, deploy)
    one_off_mappings = agh.build_oneoff_mappings(age_le)
    one_off = OneOffAccuracy(one_off_mappings)
    custom_objects={'one_off_accuracy': one_off.one_off_accuracy}
    age_model = load_model(age_path, custom_objects=custom_objects)

    # initialize image preprocessors
    sp = SimplePreprocessor(256, 256, inter=cv2.INTER_CUBIC)
    age_mp = MeanPreprocessor(age_mean['R'], age_mean['G'], age_mean['B'])
    gender_mp = MeanPreprocessor(gender_mean['R'], gender_mean['G'], gender_mean['B'])
    cp = CropPreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE)
    iap = ImageToArrayPreprocessor()

    # initialize dlib's face detector (HOG-based), then create facial landmark predictor and face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
    fa = FaceAligner(predictor)

    for image_path in np.array(image_paths)[rows]:
        path = os.path.sep.join([config.BASE_PATH, '*', f'{image_path}'])
        path = glob.glob(path)[0]
        # load image fron disk, resize it and convert it to grayscale
        print(f'[INFO] processing {path}')
        image = cv2.imread(path)
        image = imutils.resize(image, width=1024)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in grayscale image
        rects = detector(gray, 1)

        # loop over face detections
        for rect in rects:
            # determine facial landmarks for face region, then align face
            shape = predictor(gray, rect)
            face = fa.align(image, gray, rect)

            age_preds, gender_preds = prediction(config, face, sp, age_mp, gender_mp, cp, age_model, gender_model)

            # visualize age and gender predictions
            age_canvas = agh.visualize_age(age_preds, age_le)
            gender_canvas = agh.visualize_gender(gender_preds, gender_le)

            # draw bounding box around face
            clone = image.copy()
            x, y, w, h = face_utils.rect_to_bb(rect)
            cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # show output iamge
            cv2.imshow('Input', clone)
            cv2.imshow('Face', face)
            cv2.imshow('Age Probabilities', age_canvas)
            cv2.imshow('Gender Probabilities', gender_canvas)
            cv2.waitKey(0)
            
elif 'ADIENCE' in config.DATASET:
    size_per_cross = args['sample_size']//config.NUM_FOLD_PATHS
    # c = 0
    for i in range(config.NUM_FOLD_PATHS):
        # load Label Encoder and mean files
        print(f'[INFO] loading label encoders and mean files for cross validation {i}...')

        age_le = pickle.loads(open(deploy.AGE_LABEL_ENCODERS[i], 'rb').read())
        gender_le = pickle.loads(open(deploy.GENDER_LABEL_ENCODERS[i], 'rb').read())
        age_mean = json.loads(open(deploy.AGE_MEANS[i]).read())
        gender_mean = json.loads(open(deploy.GENDER_MEANS[i]).read())

        # load model from disk
        age_path = deploy.AGE_NETWORK_PATHS[i]
        gender_path = deploy.GENDER_NETWORK_PATHS[i]
        gender_model = load_model(gender_path)

        agh = AgeGenderHelper(config, deploy)
        one_off_mappings = agh.build_oneoff_mappings(age_le)
        one_off = OneOffAccuracy(one_off_mappings)
        custom_objects={'one_off_accuracy': one_off.one_off_accuracy}
        age_model = load_model(age_path, custom_objects=custom_objects)

        # initialize image preprocessors
        sp = SimplePreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE, inter=cv2.INTER_CUBIC)
        age_mp = MeanPreprocessor(age_mean['R'], age_mean['G'], age_mean['B'])
        gender_mp = MeanPreprocessor(gender_mean['R'], gender_mean['G'], gender_mean['B'])
        cp = CropPreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE, inter=cv2.INTER_CUBIC)
        iap = ImageToArrayPreprocessor()

        # load testing hdf5 file
        test_db_age = h5py.File(deploy.AGE_HDF5S[i])
        test_db_gender = h5py.File(deploy.GENDER_HDF5S[i])
        rows = np.random.randint(low=0, high=len(test_db_age['labels']), size=size_per_cross)

        # loop over rows
        for i, row in enumerate(rows):
            # load image
            gt_label_age, image = test_db_age['labels'][row], test_db_age['images'][row]
            clone = image.copy()

            age_preds, gender_preds = prediction(config, clone, sp, age_mp, gender_mp, cp, age_model, gender_model)

            # visualize age and gender predictions
            age_canvas = agh.visualize_age(age_preds, age_le)
            gender_canvas = agh.visualize_gender(gender_preds, gender_le)
            image = imutils.resize(image, width=400)

            # draw actual prediction on the image
            gt_label_age = age_le.classes_[gt_label_age].split('_')
            cv2.putText(image, 'Actual: {}-{}'.format(gt_label_age[0], gt_label_age[1]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

            # show image
            cv2.imwrite('image.jpg', image)
            image = cv2.imread('image.jpg')
            cv2.imshow('Image', image)
            cv2.imshow('Age Probabilities', age_canvas)
            cv2.imshow('Gender Probabilities', gender_canvas)
            # cv2.imwrite(f'{c:02d}_A_image.jpg', image)
            # cv2.imwrite(f'{c:02d}_Age_Probabilities.jpg', age_canvas)
            # cv2.imwrite(f'{c:02d}_Gender_Probabilities.jpg', gender_canvas)
            # c += 1
            cv2.waitKey(0)
            # os.unlink('image.jpg')