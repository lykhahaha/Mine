import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3

# Load and create datasets, train_x = [60k, 28, 28], train_y = [60k]
(train_x, train_y), (val_x, val_y) = datasets.mnist.load_data()
train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)/255.
val_x = tf.convert_to_tensor(val_x, dtype=tf.float32)/255.

train_y = tf.convert_to_tensor(train_y, dtype=tf.int32)
val_y = tf.convert_to_tensor(val_y, dtype=tf.int32)

print(tf.reduce_min(train_x), tf.reduce_max(train_x))
print(tf.reduce_min(train_y), tf.reduce_max(train_y))

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(BATCH_SIZE)

sample = next(iter(train_dataset))
print(f'[INFO] batch: {sample[0].shape}')

# [BATCH_SIZE, 784] => [BATCH_SIZE, 256] => [BATCH_SIZE, 128] => [BATCH_SIZE, 10]
w_1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b_1 = tf.Variable(tf.zeros([256]))
w_2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b_2 = tf.Variable(tf.zeros([128]))
w_3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b_3 = tf.Variable(tf.zeros([10]))

for e in range(EPOCHS):
    epoch_loss = 0
    for step, (batch_x, batch_y) in enumerate(train_dataset):
        batch_x = tf.reshape(batch_x, [-1, 28*28])

        with tf.GradientTape() as tape:
            # [BATCH_SIZE, 784]@[784, 256]+[1, 256] = [BATCH_SIZE, 256]
            h_1 = batch_x@w_1 + tf.broadcast_to(b_1, [1, 256])
            h_1 = tf.nn.relu(h_1)
            # [BATCH_SIZE, 256]@[256, 128]+[128] = [BATCH_SIZE, 128]
            h_2 = h_1@w_2 + b_2
            h_2 = tf.nn.relu(h_2)
            # [BATCH_SIZE, 128]@[128, 10]+[10] = [BATCH_SIZE, 10]
            out = h_2@w_3 + b_3

            y_onehot = tf.one_hot(batch_y, depth=10)

            loss = tf.reduce_mean(tf.square(y_onehot - out))
        
        epoch_loss += (loss*batch_x.shape[0])

        # Compute gradients
        grads = tape.gradient(loss, [w_1, b_1, w_2, b_2, w_3, b_3])
        # w_1 = w_1 - lr * w1_grad
        w_1.assign_sub(LR * grads[0])
        b_1.assign_sub(LR * grads[1])
        w_2.assign_sub(LR * grads[2])
        b_2.assign_sub(LR * grads[3])
        w_3.assign_sub(LR * grads[4])
        b_3.assign_sub(LR * grads[5])

    print(f'[INFO] Epoch {e+1}/{EPOCHS} - Loss: {epoch_loss/train_y.shape[0]:.4f}')