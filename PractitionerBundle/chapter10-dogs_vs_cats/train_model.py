# USAGE
# python train_model.py --db ../datasets/kaggle_dogs_vs_cats/hdf5/features.hdf5 --model dogs_vs_cats.pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pickle
import h5py
import argparse

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--db', required=True, help='path to HDF5 database')
ap.add_argument('-m', '--model', required=True, help='path to output model')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='# of jobs when tuning hyperparameters')
args = vars(ap.parse_args())

# open HDF5 database and partition data by determining index
db = h5py.File(args['db'], 'r')
i = int(len(db['labels'])*0.75)

# define set of hyperparameter
print('[INFO] tuning hyperparameter...')
params = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
model = GridSearchCV(LogisticRegression(), param_grid=params, cv=3, n_jobs=args['jobs'])
model.fit(db['features'][:i], db['labels'][:i])
print(f'[INFO] best hyperparameter: {model.best_params_}')

# evaluate model
print('[INFO] evaluating...')
preds = model.predict(db['features'][i:])
print(classification_report(db['labels'][i:], preds, target_names=db['label_names']))

# compute raw accuracy with extra precision
acc = accuracy_score(db['labels'][i:], preds)
print(f'[INFO] score: {acc*100:.2f}')

# save model
print('[INFO] serializing model...')
f = open(args['model'], 'wb')
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close database
db.close()