# USAGE
# python regularization.py --dataset ../datasets/animals
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
args = vars(ap.parse_args())

print('[INFO] loading images...')
img_paths = list(paths.list_images(args['dataset']))
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader([sp])
data, labels = sdl.load(img_paths, verbose=500)
data = data.reshape((len(data), -1))

# Encode the labels as integer
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split to train and test set
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

# Training on multiple types of regularization
for r in [None, 'l1', 'l2']:
    # train a SGD classifier using a softmax loss function and the
    # specified regularization function for 10 epochs
    print(f'[INFO] training model with {r} penalty')
    model = SGDClassifier(loss='log', penalty=r, max_iter=10, learning_rate='constant', eta0=0.1, random_state=42)
    model.fit(trainX, trainY)
    
    acc = model.score(testX, testY)
    print(f'[INFO] {r} penalty accuracy: {acc*100:.2f}%')