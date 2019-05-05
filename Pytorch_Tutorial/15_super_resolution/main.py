# Test image on model running in Caffe2
# USAGE
# python main.py --image-path cat.jpg --result-path super_cat.jpg --init-path init_net.pb --predict-path predict_net.pb
from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils
import numpy as np
from os import path
import subprocess
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
import argparse

IMAGE_SIZE = 224

# Construct argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image-path', required=True, help='path to image to test')
ap.add_argument('-r', '--result-path', required=True, help='path to result image')
ap.add_argument('-ip', '--init-path', required=True, help='path to init_net file')
ap.add_argument('-pp', '--predict-path', required=True, help='path to predict_net file')
args = vars(ap.parse_args())

# Load caffe2 model
init_net = open(args['init_path'], 'rb').read()
predict_net = open(args['predict_path'], 'rb').read()

# Load image
image = io.imread(args['image_path'])
# resize image to desired dimensions
image = transform.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

# convert image to Ybr format
image = Image.fromarray(image.astype('uint8'))
image = image.convert('YCbCr')
image_y, image_cb, image_cr = image.split()

# Run mobile nets that are already generated  so that caffe2 workspace is properly initialized
workspace.RunNetOnce(init_net)
workspace.RunNetOnce(predict_net)

# inspect what network looks like and identify what input and output blob names are
print('[INFO] Caffe2 network...')
print(net_printer.to_string(predict_net))

# From the above output, we can see that input is named “9” and output is named “27”
workspace.FeedBlob('9', np.array(image_y)[np.newaxis, np.newaxis, :, :].astype(np.float32))

# run preidct_net to get model output
workspace.RunNetOnce(predict_net)

# get model output blob
image_out = workspace.FetchBlob('27')

# Now, refer back to the post-processing steps in PyTorch implementation of super-resolution model here to construct back the final output image and save the image.
image_out_y = Image.fromarray(np.uint8((image_out[0, 0]).clip(0, 255)), mode='L')

# Get the output image follow post-processing step from PyTorch implementation
final_image = Image.merge(
    'YCbCr',
    [
        image_out_y,
        image_cb.resize(image_out_y.size, Image.BICUBIC),
        image_cr.resize(image_out_y.size, Image.BICUBIC)
    ]
).convert('RGB')

# Save final image
final_image.save(args['result_path'])