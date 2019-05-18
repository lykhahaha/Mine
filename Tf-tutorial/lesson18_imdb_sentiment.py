import tensorflow as tf
import numpy as np
from tensorflow import keras as K
from tensorflow.keras import datasets, optimizers, preprocessing, Model
from tensorflow.keras.layers import LSTM, Embedding, Dense
import os

UNITS = 64
NUM_CLASSES = 2
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20

tf.random.set_seed(42)
np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# load dataset but only keep top n words, zero the rest
top_words = 10000
# truncate and pad input sentences
max_review_length = 80
(train_x, train_y), (val_x, val_y) = datasets.imdb.load_data(num_words=top_words) # numpy 1.16.2

train_x = preprocessing.sequence.pad_sequences(train_x, maxlen=max_review_length)
val_x = preprocessing.sequence.pad_sequences(val_x, maxlen=max_review_length)

class RNN(Model):
    def __init__(self, units, num_classes, num_layers):
        super().__init__()

        self.rnn = LSTM(units, return_sequences=True)
        self.rnn2 = LSTM(units)

        # have 10000 words totally, every word will be embedding into 100 length vector the max sentence length is 80 words
        self.embedding = Embedding(top_words, 100, input_length=max_review_length)
        self.fc = Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)

        x = self.rnn(x)
        x = self.rnn2(x)

        x = self.fc(x)

        return x

def main():
    model = RNN(UNITS, NUM_CLASSES, num_layers=2)

    model.compile(optimizer=optimizers.Adam(lr=LR),
                loss = K.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2)

if __name__ == '__main__':
    main()