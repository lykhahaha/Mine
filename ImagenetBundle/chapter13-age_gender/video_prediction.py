# USAGE
# python video_prediction.py --image image.jpg
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
ap.add_argument('-v', '--video', help='(optional) path to video file')
args = vars(ap.parse_args())

# if video path is not supplied, refer to webcam, otherwise load video
if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

# load Label Encoder and mean files
print('[INFO] loading label encoders and mean files...')
age_le = pickle.loads(open(deploy.AGE_LABEL_ENCODER, 'rb').read())
gender_le = pickle.loads(open(deploy.GENDER_LABEL_ENCODER, 'rb').read())
age_means = json.loads(open(deploy.AGE_MEANS).read())
gender_means = json.loads(open(deploy.GENDER_MEANS).read())

# load model from disk
custom_objects = None

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
age_mp = MeanPreprocessor(age_means['R'], age_means['G'], age_means['B'])
gender_mp = MeanPreprocessor(gender_means['R'], gender_means['G'], gender_means['B'])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

# initialize dlib's face detector (HOG-based), then create facial landmark predictor and face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(deploy.DLIB_LANDMARK_PATH)
fa = FaceAligner(predictor)

# keep looping
while True:
    # get the current frame
    grabbed, frame = camera.read()

    # if we are viewing a video and we did not grab a frame, we reach the end of video
    if args.get('video') and not grabbed:
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clone = frame.copy()

    # detect faces in grayscale image
    rects = detector(gray, 1)
    
    # loop over face detections
    for rect in rects:
        # align faces
        face = fa.align(frame, gray, rect)

        # resize and crop image
        age_crops = cp.preprocess(age_mp.preprocess(sp.preprocess(face)))
        age_crops = np.array([iap.preprocess(c) for c in age_crops])

        gender_crops = cp.preprocess(gender_mp.preprocess(sp.preprocess(face)))
        gender_crops = np.array([iap.preprocess(c) for c in gender_crops])

        # predict on age and gender based on extracted crops
        age_pred = age_model.predict(age_crops).mean(axis=0)
        gender_pred = gender_model.predict(gender_crops).mean(axis=0)

        # draw bounding box around face
        x, y, w, h = face_utils.rect_to_bb(rect)
        cv2.rectangle(clone, (x, y), (x+w, y+h), (0, 255, 0), 2)

        clone = agh.visualize_video(age_pred, gender_pred, age_le, gender_le, clone, (x, y))

    cv2.imshow('Output', clone)
    
    # if 'q' is pressed, stop loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup camera and close any open windows
camera.release()
cv2.destroyAllWindows()