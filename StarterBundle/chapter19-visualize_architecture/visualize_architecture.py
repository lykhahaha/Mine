from pyimagesearch.nn.conv import MiniVGGNet
from keras.utils import plot_model
import os

# conda install python-graphviz
# Go to D:\Anaconda\envs\py36\Lib\site-packages\keras\engine\sequential.py and comment out def in line 95
os.environ["PATH"] += os.pathsep + "E:/graphviz/bin/"

# initialize LeNet and then write network architecture visualization graph to disk
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
plot_model(model, to_file='miniVGG.png', show_shapes=True)