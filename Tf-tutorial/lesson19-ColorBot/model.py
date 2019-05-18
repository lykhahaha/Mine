import tensorflow as tf
import numpy as np
from tensorflow import keras as K
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTMCell, Embedding, Dense

class RNNColorBot(Model):
    """
    Multi-layer (LSTM) RNN 
    """
    def __init__(self, rnn_cell_sizes, label_dim, keep_prob):
        """
        Construct RNNColorBot
        rnn_cell_sizes: list of integers denoting the size of each LSTM cell in the RNN; rnn_cell_sizes[i] is the size of the i-th layer cell
        label_dimension: the length of the labels
        """
        super()__init__(name='')
        self.rnn_cell_sizes = rnn_cell_sizes
        self.label_dimension = label_dim
        self.keep_prob = keep_prob

        self.cells = [LSTMCell(size) for size in self.rnn_cell_sizes]
        self.relu = Dense(self.label_dimension, activation=tf.nn.relu)

    def call(self, inputs, training=None):
        """
        inputs: A tuple (characters, sequence_length), characters: [batch_size, time_steps, 256], sequence_length: the length of each character sequence
        Return:
        A tensor of dimension [batch_size, label_dimension]
        """
        (characters, sequence_length) = inputs
        # Transpose the first and second dimensions so that chars is of shape [time_steps, batch_size, dimension].
        characters = tf.transpose(characters, [1, 0, 2])
        