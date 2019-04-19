from keras.callbacks import Callback
import os

class EpochCheckpoint(Callback):
    def __init__(self, output_path, every=5, start_at=0):
        super().__init__()

        self.output_path = output_path
        self.every = every
        self.int_epoch = start_at

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model should be serialized to disk
        if (self.int_epoch + 1) % self.every == 0:
            p = os.path.sep.join([self.output_path, f'epoch_{self.int_epoch + 1}.hdf5'])
            self.model.save(p, overwrite=True)
        
        # increase internal epoch counter
        self.int_epoch += 1