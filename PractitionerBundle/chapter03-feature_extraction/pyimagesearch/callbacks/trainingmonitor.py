from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path, json_path=None, start_at=0):
        # store the output path for the figure, the path to the JSON serialized file and starting epoch
        super(TrainingMonitor, self).__init__()
        # The path to the output plot that we can use to visualize loss and accuracy over time.
        self.fig_path = fig_path
        # An optional path used to serialize the loss and accuracy values as a JSON file
        self.json_path = json_path 
        # This is the starting epoch that training is resumed at when using ctrl + c training.
        self.start_at = start_at 
    
    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                # check to see if a training epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and trim any entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def on_epoch_end(self, epoch, logs={}):

        # loop over the logs and update the loss, accuracy, etc. for the entire training process
        for k, v in logs.items():
            l = self.H.get(k, [])
            if isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            elif isinstance(v, np.ndarray):
                v = v.tolist()
            l.append(v)

            self.H[k] = l
        
        # check to see if training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, 'w')
            f.write(json.dumps(self.H))
            f.close()

        # ensure at least 2 epochs have passed before plotting (epoch starts at 0)
        if len(self.H['loss']) > 1:
            # plot training loss and accuracy
            plt.style.use('ggplot')
            plt.rcParams["figure.figsize"] = (9.0, 7.2)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            color_legend_iter = iter(('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'))
            for key in self.H.keys():
                if key == 'lr':
                    continue
                if 'acc' in key.split('_')[-1]:
                    ax1.plot(self.H[key], label=key, color=next(color_legend_iter))
                elif 'loss' in key.split('_')[-1]:
                    ax2.plot(self.H[key], label=key, color=next(color_legend_iter))

            plt.title(f"Training Loss and Accuracy [Epoch {len(self.H['loss'])}]")
            ax1.set_xlabel('Epoch #')
            ax1.set_ylabel('Accuracy')
            ax2.set_ylabel('Loss')
            fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.07), ncol=len(self.H.keys())//2, fancybox=True, framealpha=0)
            plt.savefig(self.fig_path)
            plt.close()