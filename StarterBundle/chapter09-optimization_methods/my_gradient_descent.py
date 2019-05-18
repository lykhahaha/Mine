# USAGE
# python my_gradient_descent.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    return 1./(1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds


# Construct argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=int, default=100, help='# of epochs')
ap.add_argument('-a', '--alpha', type=int, default=0.01, help='learning rate')
args = vars(ap.parse_args())

# generate 1000 data points, X.shape=(1000, 2), y.shape=(1000)
X, y = make_blobs(n_samples=1000, n_features=2, centers=2,
                  cluster_std=1.5, random_state=42)
y = y.reshape((len(y), 1))

# insert column of 1's
X = np.c_[X, np.ones((len(X)))]

# partition dataset
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.5, random_state=42)

print('[INFO] training...')
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over desired number of epochs
for e in range(args['epochs']):
    preds = sigmoid_activation(train_X.dot(W))

    err = preds - train_y
    loss = np.sum(err**2)
    losses.append(loss)

    d = err * sigmoid_deriv(preds)
    gradient = train_X.T.dot(d)
    W -= args['alpha'] * gradient

    if (e + 1) % 5 == 0:
        print(f"[INFO] Epoch {e+1}/{args['epochs']} - loss: {loss}")

# evaluate model
print('[INFO] evaluating...')
preds = predict(test_X, W)
print(classification_report(test_y, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(test_X[:, 0], test_X[:, 1], marker="o", c=test_y[:, 0], s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
