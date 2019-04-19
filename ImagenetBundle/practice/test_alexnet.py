from config import imagenet_alexnet_config as config
from pyimagesearch.nn.mxconv import MxAlexNet
import mxnet as mx
import json
import argparse
import os

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-p', '--prefix', required=True, help='name of model prefix')
args = vars(ap.parse_args())

# define testing set iterator
test_iter = mx.io.ImageRecordIter(path_imgrec=config.TEST_MX_REC, data_shape=(3, 227, 227), batch_size=config.BATCH_SIZE, mean_r=means['R'], mean_g=means['G'], mean_b=means['B'])

# load R, G, B means
means = json.loads(open(config.DATASET_MEAN).read())

# checkpoints path
checkpoints_path = os.path.sep.join([args['checkpoints'], args['prefix']])

# load model
model = mx.model.FeedForward.load(prefix=checkpoints_path, epoch=args['epoch'])
model = mx.model.FeedForward(symbol=model.symbol, ctx=[mx.gpu(0)], arg_params=model.arg_params, aux_params=model.aux_params)

metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5)]
rank1, rank5 = model.score(test_iter, eval_metric=metrics)