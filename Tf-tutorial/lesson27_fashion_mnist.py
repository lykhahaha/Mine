import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 30

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(train_x, train_y), (val_x, val_y) = datasets.fashion_mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.map(preprocess).shuffle(10000).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.map(preprocess).batch(BATCH_SIZE)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(21, activation=tf.nn.relu),
    layers.Dense(10)
])

model.build(input_shape=[None, 28*28])
model.summary()

opt = optimizers.Adam(lr=LR)

def main():
    for e in range(EPOCHS):
        train_epoch_loss = 0
        train_total_num, val_total_num = 0, 0
        val_correct = 0
        for batch_x, batch_y in train_dataset:
            batch_x = tf.reshape(batch_x, [-1, 28*28])

            with tf.GradientTape() as tape:
                logits = model(batch_x)
                batch_y_onehot = tf.one_hot(batch_y, depth=10)

                loss_mse = tf.reduce_mean(tf.losses.MSE(batch_y_onehot, logits))
                loss_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(batch_y_onehot, logits, from_logits=True))
            train_epoch_loss += (loss_ce.numpy() * batch_x.shape[0])
            train_total_num += batch_x.shape[0]

            grads = tape.gradient(loss_ce, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

        for batch_x, batch_y in val_dataset:
            batch_x = tf.reshape(batch_x, [-1, 28*28])

            logits = model(batch_x)
            prob = tf.nn.softmax(logits, axis=1)

            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)

            correct = tf.equal(pred, batch_y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            val_correct += correct.numpy()
            val_total_num += batch_x.shape[0]
        
        print(f'[INFO] Epoch {e+1}/{EPOCHS} - Loss: {train_epoch_loss/train_total_num:.4f} - Val acc: {val_correct/val_total_num:.4f}')

if __name__ == '__main__':
    main()