from keras import backend as K
import tensorflow as tf
import numpy as np

class OneOffAccuracy:
    def __init__(self, one_off_mappings):
        self.one_off_mappings = one_off_mappings

    def one_off_accuracy(self, labels, preds):
        return tf.py_func(self.one_off_compute, [labels, preds], tf.float32)

    def one_off_compute(self, labels, preds):
        # initialize one-off accuracy and get argmax from labels and preds
        one_off = 0
        labels_argmax, preds_argmax = labels.argmax(axis=1), preds.argmax(axis=1)
        
        # check to see if pred is in label one-off mapping
        for label_argmax, pred_argmax in zip(labels_argmax, preds_argmax):
            if pred_argmax in self.one_off_mappings[label_argmax]:
                one_off += 1
        
        one_off /= float(len(labels))
        return np.float32(one_off)