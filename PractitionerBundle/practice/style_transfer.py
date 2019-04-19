from pyimagesearch.nn.conv import NeuralStyle
from keras.applications import VGG19

SETTINGS = {
    'input_path': 'inputs/quan.jpg',
    'style_path': 'inputs/red_chalk.jpg',
    'output_path': 'output',

    'net': VGG19,

    'content_layer': 'block4_conv2',
    'style_layer': ['block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4'],

    'content_weight': 1.0,
    'style_weight': 100.0,
    'tv_weight': 10.0,

    'iterations': 50
}

nn = NeuralStyle(SETTINGS)
nn.transfer()