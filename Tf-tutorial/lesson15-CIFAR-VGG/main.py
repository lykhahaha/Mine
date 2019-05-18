import tensorflow as tf
from tensorflow import keras as K
import numpy as np
from tensorflow.keras import datasets, optimizers
from network import VGG16

BATCH_SIZE = 128
LR = 1e-4
EPOCHS = 250

def normalize(train_x, val_x):
    """
    normalize inputs for zero mean and unit variance
    """
    mean  = np.mean(train_x)
    std = np.std(train_x)
    print(f'[INFO] Mean of training part: {mean:.4f}, std of training part: {std:.4f}')
    train_x = (train_x - mean) / (std + 1e-7)
    val_x = (val_x - mean) / (std + 1e-7)
    return train_x, val_x

def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x, y

def main():
    tf.random.set_seed(42)

    print('[INFO] loading data...')
    (train_x, train_y), (val_x, val_y) = datasets.cifar10.load_data()
    train_x, val_x = normalize(train_x, val_x)
    train_loader = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_loader = train_loader.map(preprocess).shuffle(50000).batch(BATCH_SIZE)
    val_loader = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_loader = train_loader.map(preprocess).batch(BATCH_SIZE)

    model = VGG16([32, 32, 3])

    criterion = K.losses.CategoricalCrossentropy(from_logits=True)
    criterion_save = K.metrics.CategoricalCrossentropy(from_logits=True)
    metric = K.metrics.CategoricalAccuracy()
    opt = optimizers.Adam(lr=LR)

    for e in range(EPOCHS):
        for batch_x, batch_y in train_loader:
            # [b, 1] => [b]
            batch_y = tf.squeeze(batch_y, axis=1)
            # [b, 10]
            batch_y = tf.one_hot(batch_y, depth=10)

            with tf.GradientTape() as tape:
                logits = model(batch_x)
                loss = criterion(batch_y, logits)
                criterion_save.update_state(batch_y, logits)
                metric.update_state(batch_y, logits)
            
            grads = tape.gradient(loss, model.trainable_variables)
            # MUST clip gradient here or it will disverge
            grads = [tf.clip_by_norm(g, 15) for g in grads]
            opt.apply_gradients(zip(grads, model.trainable_variables))

        train_loss, acc = criterion_save.result().numpy(), metric.result().numpy()

        criterion_save.reset_states()
        metric.reset_states()

        for batch_x, batch_y in val_loader:
            # [b, 1] => [b]
            batch_y = tf.squeeze(batch_y, axis=1)
            # [b, 10]
            batch_y = tf.one_hot(batch_y, depth=10)

            logits = model.predict(batch_x)

            criterion_save.update_state(batch_y, logits)
            metric.update_state(batch_y, logits)
        
        print(f'[INFO] Epoch {e+1}/{EPOCHS} - loss: {train_loss:.4f} - accuracy: {acc:.4f} - val loss: {criterion_save.result().numpy():.4f} - val acc: {metric.result().numpy():.4f}')

        criterion_save.reset_states()
        metric.reset_states()

if __name__ == '__main__':
    main()