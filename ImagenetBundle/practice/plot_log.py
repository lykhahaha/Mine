import matplotlib.pyplot as plt
import re
import argparse
import numpy as np

# construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', required=True, help='name of the network')
ap.add_argument('-d', '--dataset', required=True, help='name of the dataset')
args = vars(ap.parse_args())

# INFO:root:Epoch[3] Batch [500]	Speed: 1997.40 samples/sec	accuracy=0.013391	top_k_accuracy_5=0.048828	cross-entropy=6.878449
# INFO:root:Epoch[73] Resetting Data Iterator
# INFO:root:Epoch[73] Time cost=728.322
# INFO:root:Saved checkpoint to "imagenet/checkpoints/alexnet-0074.params"
# INFO:root:Epoch[73] Validation-accuracy=0.559794
# INFO:root:Epoch[73] Validation-top_k_accuracy_5=0.790751
# INFO:root:Epoch[73] Validation-cross-entropy=1.914535

logs = [
    (65, 'training_0.log'),
    (85, 'training_65.log'),
    (100, training_85.log)
]

train_rank1, train_rank5, train_loss = [], [], []
val_rank1, val_rank5, val_loss = [], [], []

for i, (epoch_end, log_file) in enumerate(logs):
    # open log file
    rows = open(log_file).read().strip()
    epochs = set(re.findall(r'Epoch\[(\d+)\]', rows))
    epochs = sorted(int(epoch) for epoch in epochs)

    batch_end_train_rank1, batch_end_train_rank5, batch_end_train_loss = [], [], []
    batch_end_val_rank1, batch_end_val_rank5, batch_end_val_loss = [], [], []

    for epoch in epochs:
        s = r'Epoch\[' + str(epoch) + '\].*accuracy=(0*\.?[0-9]+)'
        rank1 = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(epoch) + '\].*top_k_accuracy_5=(0*\.?[0-9]+)'
        rank5 = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(epoch) + '\].*cross-entropy=(0*\.?[0-9]+)'
        loss = re.findall(s, rows)[-2]

        batch_end_train_rank1.append(float(rank1))
        batch_end_train_rank5.append(float(rank5))
        batch_end_train_loss.append(float(loss))

    batch_end_val_rank1 = re.findall(r'Validation-accuracy=(.*)', rows)
    batch_end_val_rank5 = re.findall(r'Validation-top_k_accuracy_5=(.*)', rows)
    batch_end_val_loss = re.findall(r'Validation-cross-entropy=(.*)', rows)

    batch_end_val_rank1 = [float(x) for x in batch_end_val_rank1]
    batch_end_val_rank5 = [float(x) for x in batch_end_val_rank5]
    batch_end_val_loss = [float(x) for x in batch_end_val_loss]

    train_end = epoch_end
    val_end = epoch_end

    if i > 0:
        train_end = epoch_end - logs[i-1][0]
        val_end = epoch_end - logs[i-1][0]

    train_rank1.extend(batch_end_train_rank1[0:train_end])
    train_rank5.extend(batch_end_train_rank5[0:train_end])
    train_loss.extend(batch_end_train_loss[0:train_end])

    val_rank1.extend(batch_end_val_rank1[0:train_end])
    val_rank5.extend(batch_end_val_rank5[0:train_end])
    val_loss.extend(batch_end_val_loss[0:train_end])