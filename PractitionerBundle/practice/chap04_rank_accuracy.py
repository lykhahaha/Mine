import h5py
import pickle
from pyimagesearch.utils.ranked import rank5_accuracy
import numpy as np
import argparse

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--db', required=True, help='path to HDF5 database')
ap.add_argument('-m', '--model', required=True, help='path to estimator cpickle')
args = vars(ap.parse_args())

# open HDF5 database abd partition data by index provided that database is shuffled prior to writing to disk
print('[INFO] loading HDF5 database...')
db = h5py.File(args['db'], 'r')
i = int(len(db['labels'])*0.75)

# load model and predict
print('[INFO] loading model...')
model = pickle.loads(open(args['model'], 'rb').read())
preds = model.predict_proba(db['features'][i:])

# display rank_1 and rank_5
rank_1, rank_5 = rank5_accuracy(preds, db['labels'][i:])
print(f'[INFO] rank-1: {rank_1*100:.2f}')
print(f'[INFO] rank-5: {rank_5*100:.2f}')