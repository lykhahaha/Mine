# USAGE
# python train_alexnet.py --checkpoints checkpoints --prefix alexnet
# python train_alexnet.py --checkpoints checkpoints --prefix alexnet --start-epoch 50
from config import imagenet_alexnet_config as config
from pyimagesearch.nn.mxconv import MxAlexNet
import mxnet as mx
import argparse
import logging
import json
import os

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-p', '--prefix', required=True, help='name of model prefix')
ap.add_argument('-s', '--start-epoch', type=int, default=0, help='epoch to restart training at')
args = vars(ap.parse_args())

# set logging level and output file
logging.basicConfig(filename=f"training_{args['start_epoch']}.log", filemode='w', level=logging.DEBUG)

# load RGB means for training set, then determine batch size
means = json.loads(open(config.DATASET_MEAN).read())
batch_size = config.BATCH_SIZE * config.NUM_DEVICES

# construct training image iterator
train_iter = mx.io.ImageRecordIter(path_imgrec=config.TRAIN_MX_REC, data_shape=(3, 227, 227), preprocess_threads=config.NUM_DEVICES*2, batch_size=batch_size, rand_crop=True, max_shear_ratio=0.1, rotate=15, rand_mirror=True, mean_r=means['R'], mean_g=means['G'], mean_b=means['B'])

# construct validation image iterator
val_iter = mx.io.ImageRecordIter(path_imgrec=config.VAL_MX_REC, data_shape=(3, 227, 227), batch_size=batch_size, mean_r=means['R'], mean_g=means['G'], mean_b=means['B'])

# initialize the optimizer
opt = mx.optimizer.SGD(learning_rate=1e-2, momentum=0.9, wd=5e-4, rescale_grad=1./batch_size)

# construct checkpoints path, initialize model argument and auxiliary parameters
checkpoints_path = os.path.sep.join([args['checkpoints'], args['prefix']])
arg_params, aux_params = None, None

# if there is no specific model starting epoch supplied, initialize network
if args['start_epoch'] <= 0:
    # build LeNet architecture
    print('[INFO] building network...')
    model = MxAlexNet.build(config.NUM_CLASSES)
else:
    # load checkpoint from disk
    print(f"[INFO] loading epoch {args['start_epoch']}...")
    model = mx.model.FeedForward.load(checkpoints_path, args['start_epoch'])

    # update model and parameters
    arg_params = model.arg_params
    aux_params = model.aux_params
    model = model.symbol

# compile model
model = mx.model.FeedForward(symbol=model, ctx=[mx.gpu(i) for i in range(config.NUM_DEVICES)], num_epoch=100, optimizer=opt, initializer=mx.initializer.Xavier(), arg_params=arg_params, aux_params=aux_params, begin_epoch=args['start_epoch'])

# initialize callbacks and evaluation metrics
batch_end_cbs = [mx.callback.Speedometer(batch_size, frequent=500)]
epoch_end_cbs = [mx.callback.do_checkpoint(checkpoints_path)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

# train network
print('[INFO] training network...')
model.fit(X=train_iter, eval_data=val_iter, eval_metric=metrics, batch_end_callback=batch_end_cbs, epoch_end_callback=epoch_end_cbs)