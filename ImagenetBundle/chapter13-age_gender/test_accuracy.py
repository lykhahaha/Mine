# USAGE
# python test_accuracy.py --model-path output\age_best_weights.hdf5 -lb output\age_le.cpickle
# python test_accuracy.py --model-path output\best_weights_gender.hdf5
from config import age_gender_deploy as deploy
from config import age_gender_config as config
from pyimagesearch.utils import AgeGenderHelper
from pyimagesearch.callbacks import OneOffAccuracy
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.preprocessing import SimplePreprocessor, MeanPreprocessor, ImageToArrayPreprocessor, CropPreprocessor
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import argparse
import json
import pickle
import os
import numpy as np
import progressbar
import math
import shutil
import cv2

def calculate_score(means_path, label_encoder_path, best_weight_path, test_hdf5_path, cross_val=None, preds_cross=None, labels_cross = None, is_mapped=False):
    # load RGB means for training set
    means = json.loads(open(means_path).read())

    # load LabelEncoder
    le = pickle.loads(open(label_encoder_path, 'rb').read())

    # initialize image preprocessors
    sp, mp, cp, iap = SimplePreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE), MeanPreprocessor(means['R'], means['G'], means['B']), CropPreprocessor(config.IMAGE_SIZE, config.IMAGE_SIZE), ImageToArrayPreprocessor()

    custom_objects = None
    agh = AgeGenderHelper(config, deploy)
    if config.DATASET_TYPE == 'age':
        one_off_mappings = agh.build_oneoff_mappings(le)
        one_off = OneOffAccuracy(one_off_mappings)
        custom_objects={'one_off_accuracy': one_off.one_off_accuracy}

    # load model
    print(f'[INFO] loading {best_weight_path}...')
    model = load_model(best_weight_path, custom_objects=custom_objects)

    # initialize testing dataset generator, then predict
    if cross_val is None:
        print(f'[INFO] predicting in testing data (no crops){config.SALIENCY_INFO}...')
    else:
        print(f'[INFO] predicting in testing data (no crops) for cross validation {cross_val}{config.SALIENCY_INFO}...')

    test_gen = HDF5DatasetGenerator(test_hdf5_path, batch_size=config.BATCH_SIZE, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
    preds = model.predict_generator(test_gen.generator(), steps=test_gen.num_images//config.BATCH_SIZE)

    # compute rank-1 and one-off accuracies
    labels = to_categorical(test_gen.db['labels'][0: config.BATCH_SIZE * (test_gen.num_images//config.BATCH_SIZE)], num_classes=config.NUM_CLASSES)
    preds_mapped = preds.argmax(axis=1)

    if is_mapped == True:
        preds_mapped = agh.build_mapping_to_iog_labels()[preds_mapped]
    
    if cross_val is None:
        print('[INFO] serializing all images classified incorrectly for testing dataset...')
        prefix_path = os.path.sep.join([config.WRONG_BASE, config.DATASET_TYPE])

        agh.plot_confusion_matrix_from_data(config, labels.argmax(axis=1), preds_mapped, le=le, save_path=os.path.sep.join([config.OUTPUT_BASE, f'cm_{config.DATASET_TYPE}.png']))
    else:
        print(f'[INFO] serializing all images classified incorrectly for cross validation {cross_val} of testing dataset...')
        prefix_path = os.path.sep.join([config.WRONG_BASE, f'Cross{cross_val}', config.DATASET_TYPE])

        preds_cross.extend(preds_mapped.tolist())
        labels_cross.extend(labels.argmax(axis=1).tolist())

    if os.path.exists(prefix_path):
        shutil.rmtree(prefix_path)
    os.makedirs(prefix_path)

    for i, (pred, label) in enumerate(zip(preds_mapped, labels.argmax(axis=1))):
        if pred != label:
            image = test_gen.db['images'][i]

            if config.DATASET_TYPE == 'age':
                real_label, real_pred = le.classes_[label], le.classes_[pred]
                real_label = real_label.replace('_', '-')
                real_label = real_label.replace('-inf', '+')

                real_pred = real_pred.replace('_', '-')
                real_pred = real_pred.replace('-inf', '+')
            
            elif config.DATASET_TYPE == 'gender':
                real_label = 'Male' if label == 0 else 'Female'
                real_pred = 'Male' if pred == 0 else 'Female'

            cv2.putText(image, f'Actual: {real_label}, Predict: {real_pred}', (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            cv2.imwrite(os.path.sep.join([prefix_path, f'{i:05d}.jpg']), image)

    score = accuracy_score(labels.argmax(axis=1), preds_mapped)
    print(f'[INFO] rank-1: {score:.4f}')
    score_one_off = None
    if config.DATASET_TYPE == 'age':
        score_one_off = one_off.one_off_compute(labels, to_categorical(preds_mapped, num_classes=config.NUM_CLASSES))
        print(f'[INFO] one-off: {score_one_off:.4f}')
    test_gen.close()

    # re-initialize testing generator, now excluding SimplePreprocessor
    test_gen = HDF5DatasetGenerator(test_hdf5_path, config.BATCH_SIZE, preprocessors=[mp], classes=config.NUM_CLASSES)
    preds = []

    labels = to_categorical(test_gen.db['labels'], num_classes=config.NUM_CLASSES)

    print('[INFO] predicting in testing data (with crops)...')
    # initialize progress bar
    widgets = ['Evaluating: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=math.ceil(test_gen.num_images/config.BATCH_SIZE), widgets=widgets).start()

    for i, (images, _) in enumerate(test_gen.generator(passes=1)):
        for image in images:
            crops = cp.preprocess(image)
            crops = np.array([iap.preprocess(c) for c in crops])

            pred = model.predict(crops)
            preds.append(pred.mean(axis=0))

        pbar.update(i)

    pbar.finish()
    test_gen.close()

    # compute rank-1 accuracy
    preds_mapped = np.argmax(preds, axis=1)
    if is_mapped == True:
        preds_mapped = agh.build_mapping_to_iog_labels()[preds_mapped]

    score_crops = accuracy_score(labels.argmax(axis=1), preds_mapped)
    print(f'[INFO] rank-1: {score_crops:.4f}')
    score_one_off_crops = None
    if config.DATASET_TYPE == 'age':
        score_one_off_crops = one_off.one_off_compute(labels, to_categorical(preds_mapped, num_classes=config.NUM_CLASSES))
        print(f'[INFO] one-off: {score_one_off_crops:.4f}')
    
    return score, score_one_off, score_crops, score_one_off_crops

def calculate_score_adience(config, is_frontal=False):
    if is_frontal == True:
        print(f'Calculating accuracy for frontalized test set...')
    else:
        print(f'Calculating accuracy for test set...')
    score_mean, score_one_off_mean, score_crops_mean, score_one_off_crops_mean = 0, 0, 0, 0
    preds_cross, labels_cross = [], []
    for i in range(config.NUM_FOLD_PATHS):
        if is_frontal == True:
            score, score_one_off, score_crops, score_one_off_crops = calculate_score(config.DATASET_MEANS[i], config.LABEL_ENCODER_PATHS[i], config.BEST_WEIGHTS[i], config.TEST_HDF5S_FRONTAL[i], cross_val=i, preds_cross=preds_cross, labels_cross=labels_cross)
        else:
            score, score_one_off, score_crops, score_one_off_crops = calculate_score(config.DATASET_MEANS[i], config.LABEL_ENCODER_PATHS[i], config.BEST_WEIGHTS[i], config.TEST_HDF5S[i], cross_val=i, preds_cross=preds_cross, labels_cross=labels_cross)
        
        score_mean += score
        score_crops_mean += score_crops

        if config.DATASET_TYPE == 'age':
            score_one_off_mean += score_one_off
            score_one_off_crops_mean += score_one_off_crops
    
    # load LabelEncoder
    le = pickle.loads(open(config.LABEL_ENCODER_PATHS[0], 'rb').read())
    cm_path = f'cm_{config.DATASET_TYPE}_frontal.png' if is_frontal == True else f'cm_{config.DATASET_TYPE}.png'
    AgeGenderHelper.plot_confusion_matrix_from_data(config, labels_cross, preds_cross, le, save_path=os.path.sep.join([config.OUTPUT_BASE, cm_path]))
    

    print(f'[INFO] rank 1 across {config.NUM_FOLD_PATHS} validations: {score_mean/config.NUM_FOLD_PATHS:.4f}')
    print(f'[INFO] rank 1 across {config.NUM_FOLD_PATHS} validations with crops: {score_crops_mean/config.NUM_FOLD_PATHS:.4f}')

    if config.DATASET_TYPE == 'age':
        print(f'[INFO] one-off across {config.NUM_FOLD_PATHS} validations: {score_one_off_mean/config.NUM_FOLD_PATHS:.4f}')
        print(f'[INFO] one-off across {config.NUM_FOLD_PATHS} validations with crops: {score_one_off_crops_mean/config.NUM_FOLD_PATHS:.4f}')

if config.DATASET == 'IOG':
    calculate_score(config.DATASET_MEAN, config.LABEL_ENCODER_PATH, config.BEST_WEIGHT, config.TEST_HDF5)

elif config.DATASET == 'ADIENCE':
    calculate_score_adience(config)
    calculate_score_adience(config, is_frontal=True)