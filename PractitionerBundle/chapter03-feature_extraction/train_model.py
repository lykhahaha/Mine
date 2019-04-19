# USAGE
# python train_model.py --db ../datasets/animals/hdf5/features.hdf5 --model animals.cpickle
# python train_model.py --db ../datasets/caltech-101/hdf5/features.hdf5 --model caltech-101.cpickle
# python train_model.py --db ../datasets/flowers17/hdf5/features.hdf5 --model flowers17.cpickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

#  construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--db', required=True, help='path HDF5 database')
ap.add_argument('-m', '--model', required=True, help='path to output model')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='# of jobs to run when tuning hyperparameters')
args = vars(ap.parse_args())

# open HDF5 database for reading then determine index of training and test split, assume this data was already shuffled *prior* to writing it to disk
db = h5py.File(args['db'], 'r')
# Because HDF5 is too large to fit into memory, we need to determine index i
i = int(db['labels'].shape[0] * 0.75)

# define set of parameters that we want to tune then start gid search where we evaluate our model for each value of C
print('[INFO] tuning hyperparameters...')
params = {'C': [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=args['jobs'])
model.fit(db['features'][:i], db['labels'][:i])
print(f'[INFO] best hyperparameters: {model.best_params_}')# {'C': 0.1}
print(f'[INFO] best estimators: {model.best_estimator_}')# LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

# load sklearn model from cPickle
# from sklearn.externals import joblib
# load_model = joblib.load('animals.cpickle')

# evaluate model
print('[INFO] evaluating model...')
preds = model.predict(db['features'][i:])
print(classification_report(db['labels'][i:], preds, target_names=db['label_names']))

# serialize model to disk
print('[INFO] saving model...')
f = open(args['model'], 'wb')
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close database
db.close()