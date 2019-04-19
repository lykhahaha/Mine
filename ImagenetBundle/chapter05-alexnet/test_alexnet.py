# USAGE
# # python train_alexnet.py --checkpoints checkpoints --prefix alexnet --epoch 50
from config import imagenet_alexnet_config as config
import mxnet as mx
import argparse
import json
import os

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-p', '--prefix', required=True, help='name of model prefix')
ap.add_argument('-e', '--epoch', type=int, required=True, help='epoch # to load')
args = vars(ap.parse_args())

# load RGB means for training set
means = json.loads(open(config.DATASET_MEAN).read())

# construct testing image iterator
test_iter = mx.io.ImageRecordIter(path_imgrec=config.TEST_MX_REC, data_shape=(3, 227, 227), batch_size=config.BATCH_SIZE, mean_r=means['R'], mean_g=means['G'], mean_b=means['B'])

# load checkpoint from disk
print('[INFO] loading model...')
checkpoints_path = os.path.sep.join([args['checkpoints'], args['prefix']])
model = mx.model.FeedForward.load(checkpoints_path, args['epoch'])

# compile model
model = mx.model.FeedForward(symbol=model.symbol, ctx=[mx.gpu(0)], arg_params=model.arg_params, aux_params=model.arg_params)

# make predictions on testing data
print('[INFO] predicting in testing data...')
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5)]
rank1, rank5 = model.score(test_iter, eval_metric=metrics)

# display accuracies
print(f'[INFO] rank-1: {rank1*100:.2f}')
print(f'[INFO] rank-5: {rank5*100:.2f}')