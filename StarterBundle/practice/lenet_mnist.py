from pyimagesearch.nn.conv import LeNet
from keras.optimizers import SGD
from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# load mnist dataset
print('[INFO] loading mnist...')
(trainX, trainY), (testX, testY) = mnist.load_data()

# rearrange data dimensions
if K.image_data_format=='channels_first':
    trainX = trainX.reshape((len(trainX), 1, 28, 28))
    testX = testX.reshape((len(testX), 1, 28, 28))
else:
    trainX = trainX.reshape((len(trainX), 28, 28, 1))
    testX = testX.reshape((len(testX), 28, 28, 1))

# scale it to range [0, 1]
trainX = trainX.astype('float32')/255
testX = testX.astype('float32')/255
target_names = [str(x) for x in list(range(10))]

# convert label to vector
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the optimizer and network
print('[INFO] compiling the network...')
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=len(target_names))
model.compile(opt, loss='categorical_crossentropy', metrics=['accuracy'])

# training the network
print('[INFO] training the network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=2)

# evaluate the network
print('[INFO] evaluating the network...')
preds = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=target_names))

# plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='train_loss')
plt.plot(H.history['val_loss'], label='val_loss')
plt.plot(H.history['acc'], label='train_acc')
plt.plot(H.history['val_acc'], label='val_acc')
plt.xticks(np.arange(0, 21, 5))
plt.title('Training loss and accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('lenet_epoch_20.png')