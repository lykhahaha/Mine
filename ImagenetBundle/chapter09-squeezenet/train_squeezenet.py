# USAGE
# python train_vggnet.py --checkpoints checkpoints --prefix googlenet
# python train_vggnet.py --checkpoints checkpoints --prefix googlenet --start-epoch 50
from config import imagenet_vggnet_config as config
from pyimagesearch.nn.mxconv import MxSqueezeNet
import mxnet as mx
import logging
import argparse
import json
from os import path

# initialize argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--checkpoints', required=True, help='path to checkpoint directory')
ap.add_argument('-p', '--prefix', required=True, default='vgg', help='prefix of filename that mxnet creates')
ap.add_argument('-s', '--start-epoch', type=int, default=0, help='epoch number to start training from')
args = vars(ap.parse_args())

# set the logging parameters
logging.basicConfig(filename=f"training_{args['start_epoch']}.log", filemode='w', level=logging.DEBUG)

# construct checkpoint path
checkpoint_path = path.sep.join([args['checkpoints'], args['prefix']])

# get the means of dataset and initialize batch size
means = json.loads(open(config.DATASET_MEAN).read())
batch_size = config.NUM_DEVICES * config.BATCH_SIZE

# construct iterations of training and validation sets
train_iter = mx.io.ImageRecordIter(path_imgrec=config.TRAIN_MX_REC, data_shape=(3, 227, 227), preprocess_threads=config.NUM_DEVICES*2, batch_size=batch_size, rand_crop=True, max_shear_ratio=0.1, rotate=15, rand_mirror=True, mean_r=means['R'], mean_g=means['G'], mean_b=means['B'])
val_iter = mx.io.ImageRecordIter(path_imgrec=config.VAL_MX_REC, data_shape=(3, 227, 227), preprocess_threads=config.NUM_DEVICES*2, batch_size=batch_size, mean_r=means['R'], mean_g=means['G'], mean_b=means['B'])

# initialize optimizer
opt = mx.optimizer.SGD(learning_rate=1e-2, rescale_grad=1./batch_size, wd=2e-4, momentum=0.9)

# either initialize model or load model
arg_params, aux_params = None, None
if args['start_epoch'] <= 0:
    print('[INFO] building model...')
    model = MxSqueezeNet.build(config.NUM_CLASSES)
else:
    print(f"[INFO] load model from epoch {args['start_epoch']}...")
    model = mx.io.FeedForward.load(checkpoint_path, args['start_epoch'])
    arg_params, aux_params = model.arg_params, model.aux_params
    model = model.symbol

model = mx.model.FeedForward(symbol=model, ctx=[mx.gpu(i) for i in range(config.NUM_DEVICES)], num_epoch=100, optimizer=opt, initializer=mx.initializer.Xavier(), arg_params=arg_params, aux_params=aux_params, begin_epoch=args['start_epoch'])

# construct 2 callbacks and metric
batch_end_cbs = [mx.callback.Speedometer(batch_size, frequent=250)]
epoch_end_cbs = [mx.callback.do_checkpoint(checkpoint_path)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

# train model
print('[INFO] training model...')
model.fit(X=train_iter, eval_data=val_iter, eval_metric=metrics, epoch_end_callback=epoch_end_cbs, batch_end_callback=batch_end_cbs)