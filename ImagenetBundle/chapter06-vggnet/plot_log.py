import matplotlib.pyplot as plt
import numpy as np
import argparse
import re

# construct argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('-n', '--network', required=True, help='name of the network')
# ap.add_argument('-d', '--dataset', required=True, help='name of the dataset')
# args = vars(ap.parse_args())

# define paths to training logs
logs = [
    (4, 'training_0.log'),      # 1e-2
    (65, 'training_4.log'),     # 1e-2
    (89, 'training_65.log'),    # 1e-3
    (90, 'training_89.log'),    # 1e-4
]

# initialize list of train rank-1 and rank-5 accuracies, along with training loss
train_rank1, train_rank5, train_loss = [], [], []

# initialize list of validation rank-1 and rank-5 accuracies, along with validation loss
val_rank1, val_rank5, val_loss = [], [], []

# loop over training logs
for i, (end_epoch, p) in enumerate(logs):
    # INFO:root:Epoch[3] Batch [500]	Speed: 1997.40 samples/sec	accuracy=0.013391	top_k_accuracy_5=0.048828	cross-entropy=6.878449
    # INFO:root:Epoch[73] Resetting Data Iterator
    # INFO:root:Epoch[73] Time cost=728.322
    # INFO:root:Saved checkpoint to "imagenet/checkpoints/alexnet-0074.params"
    # INFO:root:Epoch[73] Validation-accuracy=0.559794
    # INFO:root:Epoch[73] Validation-top_k_accuracy_5=0.790751
    # INFO:root:Epoch[73] Validation-cross-entropy=1.914535

    # load contents of log file, then initialize batch lists for training and validation data
    rows = open(p).read().strip()
    batch_end_train_rank1, batch_end_train_rank5, batch_end_train_loss = [], [], []
    epoch_val_rank1, epoch_val_rank5, epoch_val_loss = [], [], []

    # grab set of training epochs
    epochs = set(re.findall(r'Epoch\[(\d+)\]', rows))
    epochs = sorted([int(epoch) for epoch in epochs])

    # loop over epochs
    for epoch in epochs:
        # find all rank-1 accuracies, rank-5 accuracies and loss values, then tale final entry in list for each
        s = r'Epoch\[' + str(epoch) + '\].*accuracy=([0]*\.?[0-9]+)'
        rank1 = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(epoch) + '\].*top_k_accuracy_5=([0]*\.?[0-9]+)'
        rank5 = re.findall(s, rows)[-2]
        s = r'Epoch\[' + str(epoch) + '\].*cross-entropy=([0-9]*\.?[0-9]+)'
        loss = re.findall(s, rows)[-2]

        # update batch training lists
        batch_end_train_rank1.append(float(rank1))
        batch_end_train_rank5.append(float(rank5))
        batch_end_train_loss.append(float(loss))

    # extract validation rank-1 and rank-5 accuracies for each epoch, followed by the loss
    epoch_val_rank1 = re.findall(r'Validation-accuracy=(.*)', rows)
    epoch_val_rank5 = re.findall(r'Validation-top_k_accuracy_5=(.*)', rows)
    epoch_val_loss = re.findall(r'Validation-cross-entropy=(.*)', rows)

    # convert validation rank-1, rank-5 and loss lists to floats
    epoch_val_rank1 = [float(x) for x in epoch_val_rank1]
    epoch_val_rank5 = [float(x) for x in epoch_val_rank5]
    epoch_val_loss = [float(x) for x in epoch_val_loss]

    # check to see if we are not examining first log file, if so, use the number of final epoch in log file as our slice index
    if i > 0 and end_epoch is not None:
        train_end = end_epoch - logs[i - 1][0]
        val_end = end_epoch - logs[i - 1][0]

    else:
        train_end = end_epoch
        val_end = end_epoch

    # update training lists
    train_rank1.extend(batch_end_train_rank1[0:train_end])
    train_rank5.extend(batch_end_train_rank5[0:train_end])
    train_loss.extend(batch_end_train_loss[0:train_end])

    # update validation lists
    val_rank1.extend(epoch_val_rank1[0:val_end])
    val_rank5.extend(epoch_val_rank5[0:val_end])
    val_loss.extend(epoch_val_loss[0:val_end])

# plot the accuracies and losses
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (9.0, 7.2)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
color_legend_iter = iter(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))

ax1.plot(train_rank1, label='train_rank_1', color=next(color_legend_iter))
ax1.plot(train_rank5, label='train_rank_5', color=next(color_legend_iter))
ax1.plot(val_rank1, label='val_rank_1', color=next(color_legend_iter))
ax1.plot(val_rank5, label='val_rank_5', color=next(color_legend_iter))
ax2.plot(train_loss, label='train_loss', color=next(color_legend_iter))
ax2.plot(val_loss, label='val_loss', color=next(color_legend_iter))
ax1.set_xlabel('Epoch #')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
plt.title(f"Cross-entropy Loss and Accuracy [Epoch {logs[-1][0]}]")
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.07), ncol=3, fancybox=True, framealpha=0)
plt.show()
plt.close()