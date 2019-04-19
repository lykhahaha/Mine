from keras.callbacks import Callback
import os
import json
import numpy as np

class ModelCheckpointsAdvanced(Callback):
    def __init__(self, file_path, monitor='val_acc', mode='max', json_path=None, start_at=0):
        super().__init__()
        self.file_path = file_path
        self.monitor = monitor
        self.mode = mode
        self.json_path = json_path
        self.start_at = start_at
        if self.mode == 'min':
            self.best = np.Inf
        elif self.mode == 'max':
            self.best = -np.Inf

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        history = {}
        monitor_list = []

        # if the JSON history path exists, load training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                history = json.loads(open(self.json_path).read())

                # check to see if a training epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and trim any entries that are past the starting epoch
                    monitor_list = history[self.monitor][:self.start_at]

                    if self.mode == 'min':
                        self.best = np.min(monitor_list)
                    elif self.mode == 'max':
                        self.best = np.max(monitor_list)
                    else:
                        raise AssertionError(f'{self.mode} is not either min or max')

    def on_epoch_end(self, epoch, logs={}):
        if self.monitor in logs.keys():
            if self.mode == 'min':
                if self.best > logs[self.monitor]:
                    print(f'Epoch {epoch + 1:03d}: {self.monitor} improved from {self.best:.5f} to {logs[self.monitor]:.5f}, saving model to {self.file_path}')
                    self.best = logs[self.monitor]
                    self.model.save(self.file_path, overwrite=True)
            elif self.mode == 'max':
                if self.best < logs[self.monitor]:
                    print(f'Epoch {epoch + 1:03d}: {self.monitor} improved from {self.best:.5f} to {logs[self.monitor]:.5f}, saving model to {self.file_path}')
                    self.best = logs[self.monitor]
                    self.model.save(self.file_path, overwrite=True)
            else:
                raise AssertionError(f'{self.mode} is not either min or max')
                
        else:
            raise AssertionError(f'{self.monitor} is not in {logs.keys()}')