from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import h5py
import pickle
import argparse

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--db', required=True, help='path to HDF5 database')
ap.add_argument('-m', '--model', required=True, help='path to output model')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='# of jobs when tuning hyperparameters')
args = vars(ap.parse_args())

# load HDF5 database and partition data by index provided that dataset is shuffled prior to writing to disk
print('[INFO] loading HDF5 database...')
db = h5py.File(args['db'], 'r')
i = int(len(db['labels'])*0.75)

# define set of parameters to tune and start tuning
print('[INFO] tuning hyperparameters...')
params = {'C': [0.1, 1., 10., 100., 1000., 10000.]}
model = GridSearchCV(LogisticRegression(), params, n_jobs=args['jobs'], cv=3)
model.fit(db['features'][:i], db['labels'][:i])
print(f'[INFO] best hyperparameter: {model.best_params_}')

# evaluate the network
print('[INFO] evaluating the network...')
preds = model.predict(db['features'][i:])
print(classification_report(db['labels'][i:], preds, target_names=db['label_names']))

# save model to disk
print('[INFO] saving model to disk...')
f = open(args['model'], 'wb')
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close database
db.close()