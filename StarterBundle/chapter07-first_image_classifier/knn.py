# USAGE
# python knn.py --dataset ../../SolutionDL4CV/SB_code/datasets/animals
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
ap.add_argument('-k', '--neighbors', type=int, default=1, help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1, help='# of jobs for k-NN distance (-1 uses all available cores)')
args = vars(ap.parse_args())

print('[INFO] loading images...')
img_paths = list(paths.list_images(args['dataset']))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
data, labels = sdl.load(img_paths, verbose=500)
data = data.reshape((len(data), -1))

print(f'[INFO] feature matrix: {data.nbytes/(1024*1000.0):.1f}MB')

# Encode the labels as integer
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split to train and test set
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args['jobs'])
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test),  target_names=le.classes_))