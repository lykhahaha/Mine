# USAGE
# python lenet_mnist.py
from pyimagesearch.nn.conv import LeNet
from keras.optimizers import SGD
from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

# load the mnist dataset
print('[INFO] loading MNIST dataset...')
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# apply the Keras utility function that correctly rearranges the dimensions of the image
if K.image_data_format == 'channels_first':
    train_data = train_data.reshape((len(train_data), 1, 28, 28))
    test_data = test_data.reshape((len(test_data), 1, 28, 28))
else:
    train_data = train_data.reshape((len(train_data), 28, 28, 1))
    test_data = test_data.reshape((len(test_data), 28, 28, 1))

# scale data to the range [0, 1]
train_data = train_data.astype('float32')/255
test_data = test_data.astype('float32')/255

# convert the labels from integers to vectors
le = LabelBinarizer()
train_labels = le.fit_transform(train_labels)
test_labels = le.transform(test_labels)

print(f'[INFO] Data size: {train_data.nbytes/(1024*1000):.1f}MB')

# initialize the optimizer and model
print('[INFO] compiling model...')
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=len(le.classes_))
print(f'[INFO] Number of model parameters: {model.count_params():,}')
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# train the network
print('[INFO] training network...')
H = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=128, epochs=20, verbose=2)

# evaluate the network
print('[INFO] evaluate the network...')
predictions = model.predict(test_data, batch_size=128)
print(classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

# plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='Training loss')
plt.plot(H.history['val_loss'], label='Validation loss')
plt.plot(H.history['acc'], label='Training accuracy')
plt.plot(H.history['val_acc'], label='Validation accuracy')
plt.xticks(np.arange(0, 21, 5))
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('lenet_mnist_20_epoch.png')