# Note modify C:\Users\huong\Anaconda3\lib\site-packages\torch\onnx\symbolic.py at line 1129
# def pixel_shuffle(g, self, upscale_factor):
#     dims = self.type().sizes()
#     if len(dims) != 4:
#         return _unimplemented("pixel_shuffle", "only support 4d input")
#     output_channel = dims[1] // upscale_factor // upscale_factor
#     # after_view = view(g, self, [-1, upscale_factor, upscale_factor, output_channel, dims[2], dims[3]])
#     after_view = view(g, self, [-1, output_channel, upscale_factor, upscale_factor, dims[2], dims[3]])
#     after_transpose = g.op("Transpose", after_view, perm_i=[0, 1, 4, 2, 5, 3])
#     return view(g, after_transpose,
#                 [-1, output_channel, dims[2] * upscale_factor, dims[3] *
#                  upscale_factor])

# Convert pytorch model to init_net and predict_net which are needed to run on Caffe2
# USAGE
# python convert.py --onnx-path super_resolution.onnx --init-path init_net.pb --predict-path predict_net.pb
import torch
from torch import nn
from torch.onnx import symbolic
from custompytorch.nn import SuperResolutionNet
import numpy as np
import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
from caffe2.python.predictor import mobile_exporter
from custompytorch.utils import helpers
import argparse

# Construct argument parser and parse an argument
ap = argparse.ArgumentParser()
ap.add_argument('-op', '--onnx-path', required=True, help='path to onnx file')
ap.add_argument('-ip', '--init-path', required=True, help='path to init_net file')
ap.add_argument('-pp', '--predict-path', required=True, help='path to predict_net file')
args = vars(ap.parse_args())

BATCH_SIZE = 1
IMAGE_SIZE = 224
MODEL_URL = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'

# Define Super Resolution model
print('[INFO] loading Super Resolution network...')
torch_model = SuperResolutionNet(upscale_factor=3)

# Convert Pytorch model to ONNX file
print('[INFO] converting pytorch model to onnx...')
torch_out, inputs = helpers.convert_torch2onnx(torch_model, MODEL_URL, args['onnx_path'], batch_size=BATCH_SIZE, size=IMAGE_SIZE)

# Convert ONNX model to caffe2
print('[INFO] convert onnx model to caffe2...')
prepare_backend = helpers.convert_onnx2caffe2(torch_out, inputs, args['onnx_path'])

# Extract workspace and model proto from internal representation
caffe2_workspace = prepare_backend.workspace
caffe2_model = prepare_backend.predict_net

# init_net includes model parameters and model input embedded in it
# predict_net: guide init_net execution at run-time
# Get predict_net, init_net which are needed fot running things on mobile
init_net, predict_net = mobile_exporter.Export(caffe2_workspace, caffe2_model, caffe2_model.external_input)

# Save init_net, predict_net to a file that is used to run them on mobile
with open(args['init_path'], 'wb') as f:
    f.write(init_net.SerializeToString())
with open(args['predict_path'], 'wb') as f:
    f.write(predict_net.SerializeToString())