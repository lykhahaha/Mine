# USAGE
# python grid_search.py --input-path inputs/autumn_mountain.jpg --style-path inputs/fallingwater_blueprint.jpg --output output

from pyimagesearch.nn.conv import NeuralStyle
from keras.applications import VGG19
from os import path
import os
import json
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# initialize argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-ip', '--input-path', default=True, help='path of content image')
ap.add_argument('-sp', '--style-path', default=True, help='path of style image')
ap.add_argument('-od', '--output', default=True, help='path of output directory')
args = vars(ap.parse_args())

# initialize dict to determine which points in grid search have done
completed = {}

# load completed json if existed
if path.exists('completed.json'):
    completed = json.loads(open('completed.json').read())

# initialize parameters for grid search
PARAMS = [
    'cw_1.0-sw_100.0-tvw_10.0',
    'cw_1.0-sw_1000.0-tvw_10.0',
    'cw_1.0-sw_100.0-tvw_100.0',
    'cw_1.0-sw_1000.0-tvw_1000.0',
    'cw_10.0-sw_100.0-tvw_10.0',
    'cw_10.0-sw_10.0-tvw_1000.0',
    'cw_10.0-sw_1000.0-tvw_1000.0',
    'cw_50.0-sw_10000.0-tvw_100.0',
    'cw_100.0-sw_1000.0-tvw_100.0'
]

SETTINGS = {
    # initialize path to content image, style image and output directory
    'input_path': args['input_path'],
    'style_path': args['style_path'],
    'output_path': None,

    # define CNN to be used for style transfer, along with set of content layer and style layers
    'net': VGG19,
    'content_layer': 'block4_conv2',
    'style_layers': ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],

    # store content, style and total variation weights
    'content_weight': None,
    'style_weight': None,
    'tv_weight': None,

    # number of iterations
    'iterations': 50
}

# loop over paramters in grid search
for param in PARAMS:
    # retrieve weights from string
    weights = param.split('-')
    grid = {
        'content_weight': float(weights[0].replace('cw_', '')),
        'style_weight': float(weights[1].replace('sw_', '')),
        'tv_weight': float(weights[2].replace('tvw_', ''))
    }

    # parse filenames to construct output direcotry (e.g. inputs/moutain.jpg -> mountain)
    input_filename = args['input_path'].split(path.sep)[-1]
    input_filename = input_filename[:input_filename.rfind('.')]
    style_filename = args['style_path'].split(path.sep)[-1]
    style_filename = style_filename[:style_filename.rfind('.')]

    # construct output path and create if not existed
    output_path = '_'.join([input_filename, style_filename, param])
    output_path = path.sep.join([args['output'], output_path])
    if not path.exists(output_path):
        os.makedirs(output_path)

    # complete SETTING to pass to neural style
    SETTINGS['output_path'] = output_path
    SETTINGS['content_weight'] = grid['content_weight']
    SETTINGS['style_weight'] = grid['style_weight']
    SETTINGS['tv_weight'] = grid['tv_weight']

    # construct keys used in completed json
    key = f'{input_filename}_{style_filename}_{param}'

    # check to see if key is in completed json
    if key in completed.keys():
        print(f'[INFO] skipping {key}')
        continue
    
    # perform neural style transfer with current setting
    print(f'[INFO] starting {key}...')
    nn = NeuralStyle(SETTINGS)
    nn.transfer()

    # after transferring, update completed
    completed[key] = True

# serialize json
print('[INFO] Serializing completed json...')
f = open('completed.json', 'w')
f.write(json.dumps(completed))
f.close()