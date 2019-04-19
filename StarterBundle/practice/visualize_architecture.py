from pyimagesearch.nn.conv import LeNet
from keras.utils import plot_model
import os

os.environ['PATH'] += os.pathsep + 'D:/graphviz/bin/'
model = LeNet.build(width=28, height=28, depth=1, classes=10)
plot_model(model, 'lenet.png', show_shapes=True)