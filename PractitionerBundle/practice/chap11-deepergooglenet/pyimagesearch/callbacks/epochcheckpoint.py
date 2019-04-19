from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
    def __init__(self, output_path, every, start_epoch=0):
        super().__init__()
        self.output_path = output_path
        self.every = every
        self.init_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if (self.init_epoch + 1) % self.every == 0:
            self.model.save(os.path.sep.join([self.output_path, f'epoch_{self.init_epoch + 1}.hdf5']))
        self.init_epoch += 1