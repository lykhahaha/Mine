from pyimagesearch.nn.conv import NeuralStyle
from keras.applications import VGG19

# initialize settings directory
SETTINGS = {
    # initialize path to content image, style image and path to output directory
    'input_path': 'inputs/autumn_mountain.jpg',
    'style_path': 'inputs/fallingwater_blueprint.jpg',
    'output_path': 'output',

    # define CNN to be used for style transfer, along with set of content layer and style layers
    'net': VGG19,
    'content_layer': 'block4_conv2',
    'style_layers': ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],

    # store content, style and total variation weights
    'content_weight': 1.0,
    'style_weight': 100.0,
    'tv_weight': 10.0,

    # number of iterations
    'iterations': 50
}

# perform neural style transfer
nn = NeuralStyle(SETTINGS)
nn.transfer()