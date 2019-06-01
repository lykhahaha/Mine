# USAGE
# python test_squeezenet.py.py --checkpoints checkpoints --prefix squeezenet --epoch-number 50
from config import imagenet_vggnet_config as config
import mxnet as mx
import argparse
import json
from os import path

# initialize argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to checkpoint directory')
ap.add_argument('-p', '--prefix', required=True, default='squeezenet', help='prefix of filename that mxnet creates')
ap.add_argument('-e', '--epoch-number', type=int, required=True, help='epoch number to evaluate vgg')
args = vars(ap.parse_args())

# construct checkpoint path
checkpoint_path = path.sep.join([args['checkpoints'], args['prefix']])

# get the means of dataset and initialize batch size
means = json.loads(open(config.DATASET_MEAN).read())

# construct iteration of testing set
test_iter = mx.io.ImageRecordIter(path_imgrec=config.TEST_MX_REC, data_shape=(3, 227, 227), preprocess_threads=config.NUM_DEVICES*2, batch_size=config.BATCH_SIZE, mean_r=means['R'], mean_g=means['G'], mean_b=means['B'])

# load model
print(f"[INFO] load model from epoch {args['epoch_number']}...")
model = mx.model.FeedForward.load(checkpoint_path, args['epoch_number'])
model = mx.model.FeedForward(symbol=model.symbol, ctx=[mx.gpu(i) for i in range(config.NUM_DEVICES)], arg_params=model.arg_params, aux_params=model.aux_params)

# initialize metrics
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5)]

# score model
rank_1, rank_5 = model.score(X=test_iter, eval_metric=metrics)
print(f'[INFO] rank_1: {rank_1}')
print(f'[INFO] rank_5: {rank_5}')