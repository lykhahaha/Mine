from config import imagenet_alexnet_config as config
from pyimagesearch.nn.mxconv import MxAlexNet
import mxnet as mx
import argparse
import logging
import os

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to output checkpoint directory')
ap.add_argument('-p', '--prefix', required=True, help='name of model prefix')
ap.add_argument('-s', '--start-epoch', type-int, default=0, help='epoch to restart at')
args = vars(ap.parse_args())

# set logging to serialize log
logging.basicConfig(filename=f"training_{args['start_epoch']}.log", filemode='w', level=logging.DEBUG)

# load R, G, B means
means = json.loads(open(config.DATASET_MEAN).read())
batch_size = config.BATCH_SIZE * config.NUM_DEVICES

# define train and validation iterator
train_iter = mx.io.ImageRecordIter(path_imgrec=config.TRAIN_MX_REC, data_shape=(3, 227, 227), preprocess_threads=config.NUM_DEVICES*2, batch_size=batch_size, rand_crop=True, max_shear_ratio=0.1, rotate=15, mean_r=means['R'], mean_g=means['G'], mean_b=means['B'])
val_iter = mx.io.ImageRecordIter(path_imgrec=config.VAL_MX_REC, data_shape=(3, 227, 227), batch_size=batch_size, mean_r=means['R'], mean_g=means['G'], mean_b=means['B'])

# initialize optimizer
opt = mx.optimizer.SGD(learning_rate=1e-2, momentum=0.9, wd=5e-4, rescale_grad=1./batch_size)

# initialize checkpoint and parameter
checkpoints_path = os.path.sep.join([args['checkpoints'], args['prefix']])
arg_params, aux_params = None, None

# initialize model
if args['start_epoch'] == 0:
    print('[INFO] building network...')
    model = MxAlexNet.build(config.NUM_CLASSES)
else:
    print(f"[INFO] loading epoch {args['start_epoch']...}")
    model = mx.model.FeedForward.load(prefix=checkpoints_path, epoch=args['start_epoch'])
    arg_params = model.arg_params
    aux_params = model.aux_params
    model = model.symbol

# compile model
model = mx.model.FeedForward(symbol=model, ctx=[mx.gpu(0)], num_epoch=90, initializer=mx.initializer.Xavier(), arg_params=arg_params, aux_params=aux_params, begin_epoch=args['start_epoch'])

# set callbacks
batch_end_cbs = [mx.callback.Speedometer(batch_size, frequent=500)]
epoch_end_cbs = [mx.callback.do_checkpoint(checkpoints_path)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

# train the network
model.fit(train_iter, eval_data=val_iter, eval_metric=metrics, epoch_end_callback=epoch_end_cbs, batch_end_callback=batch_end_cbs)