from os import path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential

BATCH_SIZE = 256
EPOCHS = 30

# Load and create dataset
(train_x, train_y), (val_x, val_y) = datasets.mnist.load_data()
train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)/255.
val_x = tf.convert_to_tensor(val_x, dtype=tf.float32)/255.

train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
train_y = tf.one_hot(train_y, depth=10)
val_y = tf.convert_to_tensor(val_y, dtype=tf.int32)
val_y = tf.one_hot(val_y, depth=10)

print(f'[INFO] Shape of train x and train y: {train_x.shape}, {train_y.shape}')
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

# Define sequential model
model = Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)
])
opt = optimizers.SGD(learning_rate=1e-3)

def train():
    for e in range(EPOCHS):
        train_epoch_loss, val_epoch_loss = 0, 0
        for step, (batch_x, batch_y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # reshape [BATCH_SIZE, 28, 28] -> [BATCH_SIZE, 784]
                batch_x = tf.reshape(batch_x, (-1, 28*28))
                # Compute output
                # [BATCH_SIZE, 784] -> [BATCH_SIZE, 10]
                out = model(batch_x)
                # Compute loss
                loss = tf.reduce_sum(tf.square(out - batch_y)) / batch_x.shape[0]

            # Update loss
            train_epoch_loss += loss.numpy() * batch_x.shape[0]
            # optimize and update variables
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables)) # equivalent to w' = w - lr * grad

        for step, (batch_x, batch_y) in enumerate(val_dataset):
            # reshape [BATCH_SIZE, 28, 28] -> [BATCH_SIZE, 784]
            batch_x = tf.reshape(batch_x, (-1, 28*28))
            # Compute output
            # [BATCH_SIZE, 784] -> [BATCH_SIZE, 10]
            out = model(batch_x)
            # Compute loss
            loss = tf.reduce_sum(tf.square(out - batch_y)) / batch_x.shape[0]
            # Update loss
            val_epoch_loss += loss.numpy() * batch_x.shape[0]

        print(f'[INFO] Epoch {e+1}/{EPOCHS} - Loss: {train_epoch_loss/train_y.shape[0]:.4f} - Val loss: {val_epoch_loss/val_y.shape[0]:.4f}')

if __name__ == '__main__':
    train()