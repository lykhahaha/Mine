from keras.callbacks import BaseLogger
from os import path
import json

class TrainingMonitor(BaseLogger):
    def __init__(self, json_path, fig_path=None, start_at=0):
        super().__init__()
        self.json_path = json_path
        self.fig_path = fig_path
        self.start_at = start_at
        self.H = {}

    def on_train_begin(self):
        if path.exists(self.json_path):
            if self.start_at > 0:
                records = json.loads(open(self.json_path).read())
                for key, value_list in records.items():
                    self.H[key][:self.start_at] = value_list[:self.start_at]

    def on_epoch_end(self, logs={}):
        for key, log_list in logs.keys():
            value_list = self.H.get(key, [])
            value_list.append(log_list)
            self.H[key] = value_list
            