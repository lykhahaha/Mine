import matplotlib.pyplot as plt
import re

# construct logs
logs = [
    (65, 'training_0.log'),      # 1e-2
    (85, 'training_65.log'),    # 1e-3
    (100, 'training_85.log'),    # 1e-4
]

# initialize arrays to store metrics after epoch
train_rank_1, train_rank_5, train_loss = [], [], []
val_rank_1, val_rank_5, val_loss = [], [], []

for end_epoch, log_file in logs:
    # load content of log file
    log_content = open(log_file).read().strip()
    # Obtain set of epochs appeared in log file
    epochs = set([e for e in re.findall(r'Epoch\[(\d+)\]', log_content)])
    epochs = sorted([int(e) for e in epochs])

    # loop over each epoch
    for epoch in epochs:
        # check if current epoch is out of range of epochs needed to be considered
        if epoch >= end_epoch:
            break

        # Get accuracy of training and validation from log file, append it to proper metric arrays
        pattern = r'Epoch\[' + str(epoch) + '\].*accuracy=([0]*\.?[\d]+)'
        rank_1_batch_end, val_rank_1_epoch = re.findall(pattern, log_content)[-2:]
        train_rank_1.append(float(rank_1_batch_end))
        val_rank_1.append(float(val_rank_1_epoch))

        # Get top_5 accuracy of training and validation from log file, append it to proper metric arrays
        pattern = r'Epoch\[' + str(epoch) + '\].*top_k_accuracy_5=([0]*\.?[\d]+)'
        rank_5_batch_end, val_rank_5_epoch = re.findall(pattern, log_content)[-2:]
        train_rank_5.append(float(rank_5_batch_end))
        val_rank_5.append(float(val_rank_5_epoch))

        # Get cross entropy of training and validation from log file, append it to proper metric arrays
        pattern = r'Epoch\[' + str(epoch) + '\].*cross-entropy=([\d]*\.?[\d]+)'
        loss_batch_end, val_loss_epoch = re.findall(pattern, log_content)[-2:]
        train_loss.append(float(loss_batch_end))
        val_loss.append(float(val_loss_epoch))

# plot both of accuracies and losses
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (9., 7.2)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
color_legend_iter = iter(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))

ax1.plot(train_rank_1, label='train_rank_1', color=next(color_legend_iter))
ax1.plot(train_rank_5, label='train_rank_5', color=next(color_legend_iter))
ax1.plot(val_rank_1, label='train_val_1', color=next(color_legend_iter))
ax1.plot(val_rank_5, label='train_val_5', color=next(color_legend_iter))
ax2.plot(train_loss, label='train_loss', color=next(color_legend_iter))
ax2.plot(val_loss, label='val_loss', color=next(color_legend_iter))
ax1.set_xlabel('Epoch #')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
plt.title(f"Cross-entropy Loss and Accuracy [Epoch {logs[-1][0]}]")
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.07), ncol=3, fancybox=True, framealpha=0)
plt.show()
plt.close()
