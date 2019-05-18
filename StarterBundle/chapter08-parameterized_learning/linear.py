import cv2
import numpy as np

labels = ['dog', 'cat', 'panda']
np.random.seed(2)

W = np.random.randn(3, 3072)
b = np.random.randn(3)

origin = cv2.imread('beagle.png')
image = cv2.resize(origin, (32, 32)).flatten()

scores = W.dot(image) + b

for score, label in zip(scores, labels):
    print(f'[INFO] {label}: {score:.2f}')

cv2.putText(origin, f'Label: {labels[np.argmax(scores)]}',
            (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('Image', origin)
cv2.waitKey(0)
