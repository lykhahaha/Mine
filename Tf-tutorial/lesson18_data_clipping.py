import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers

(train_x, train_y), (val_x, val_y) = datasets.mnist.load_data()
train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)/50.
train_y = tf.convert_to_tensor(train_y)
train_y = tf.one_hot(train_y, depth=10)
print(f'[INFO] Before repeating: train_x - {train_x.shape}, train_y - {train_y.shape}')

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(128).repeat(30)
batch_x, batch_y = next(iter(train_dataset))
print(f'[INFO] After repeating: train_x - {batch_x.shape}, train_y - {batch_y.shape}')

# [BATCH_SIZE, 784] => [BATCH_SIZE, 256] => [BATCH_SIZE, 128] => [BATCH_SIZE, 10]
w_1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b_1 = tf.Variable(tf.zeros([256]))
w_2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b_2 = tf.Variable(tf.zeros([128]))
w_3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b_3 = tf.Variable(tf.zeros([10]))

opt = optimizers.SGD(lr=0.01)

for step, (batch_x, batch_y) in enumerate(train_dataset):
    batch_x = tf.reshape(batch_x, (-1, 28*28))

    with tf.GradientTape() as tape:
        # [BATCH_SIZE, 784]@[784, 256]+[1, 256] = [BATCH_SIZE, 256]
        h_1 = batch_x@w_1 + tf.broadcast_to(b_1, [1, 256])
        h_1 = tf.nn.relu(h_1)
        # [BATCH_SIZE, 256]@[256, 128]+[128] = [BATCH_SIZE, 128]
        h_2 = h_1@w_2 + b_2
        h_2 = tf.nn.relu(h_2)
        # [BATCH_SIZE, 128]@[128, 10]+[10] = [BATCH_SIZE, 10]
        out = h_2@w_3 + b_3

        loss = tf.reduce_mean(tf.square(batch_y - out))
    
    # compute gradient
    grads = tape.gradient(loss, [w_1, b_1, w_2, b_2, w_3, b_3])

    if step % 100 == 0:
        for g in grads:
            print(f'[INFO] Before norm clipping: {tf.norm(g)}')

        grads, total_norm = tf.clip_by_global_norm(grads, 15)

        for g in grads:
            print(f'[INFO] After norm clipping: {tf.norm(g)}')

        print(f'[INFO] {step + 1} - loss: {loss}')

    opt.apply_gradients(zip(grads, [w_1, b_1, w_2, b_2, w_3, b_3]))