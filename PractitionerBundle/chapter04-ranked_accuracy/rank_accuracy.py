# USAGE
# python rank_accuracy.py --db ../datasets/flowers17/hdf5/features.hdf5 --estimator flowers17.cpickle
# python rank_accuracy.py --db ../datasets/caltech-101/hdf5/features.hdf5 --estimator caltech-101.cpickle
from pyimagesearch.utils.ranked import rank5_accuracy
import pickle
import argparse
import h5py

# construct argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--db', required=True, help='path HDF5 database')
ap.add_argument('-e', '--estimator', required=True, help='path to model estimator')
args = vars(ap.parse_args())

# load pre-trained model
print('[INFO] loading pretrained model...')
model = pickle.loads(open(args['estimator'], 'rb').read())

# open HDF5 database for reading then determine index of training and test split, assume that this data was shuffled *prior* to writing it to disk
db = h5py.File(args['db'], 'r')
i = int(db['labels'].shape[0] * 0.75)

# predict on test set then compute rank-1 and rank-5 accuracies
print('[INFO] predicting...')
preds = model.predict_proba(db['features'][i:])
rank1, rank5 = rank5_accuracy(preds, db['labels'][i:])

# display rank-1 and rank-5 accuracies
print(f'[INFO] rank-1: {rank1*100:.2f}%')
print(f'[INFO] rank-5: {rank5*100:.2f}%')

# close database
db.close()