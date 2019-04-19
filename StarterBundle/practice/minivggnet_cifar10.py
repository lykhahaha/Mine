from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# load dataset and scale it to range [0, 1]
print('[INFO] loading cifar10 dataset...')
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')/255
testX = testX.astype('float')/255

# convert label to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the cifar10 label names
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# initialize the optimizer and network
print('[INFO] compiling the network...')
opt = SGD(lr=0.01, momentum=0.9, decay=0.01/40, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=len(label_names))
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# training the network
print('[INFO] training the network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=100, verbose=2)

# evaluate the network
print('[INFO] evaluating the network...')
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=label_names))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='train_loss')
plt.plot(H.history['val_loss'], label='val_loss')
plt.plot(H.history['acc'], label='train_acc')
plt.plot(H.history['val_acc'], label='val_acc')
plt.title('Training loss and accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('minivgnet_cifar10_epoch_100.png')