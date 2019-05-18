import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import datetime
import matplotlib.pyplot as plt
import io
from os import path

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)

    return x, y

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image"""
    # Save the plot to a PNG in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def image_grid(images):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10,10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title='name')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
  
    return figure

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 30

(train_x, train_y), (val_x, val_y) = datasets.mnist.load_data()
train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.map(preprocess).shuffle(60000).batch(BATCH_SIZE).repeat(10)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.map(preprocess).batch(BATCH_SIZE, drop_remainder=True)

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(21, activation=tf.nn.relu),
    layers.Dense(10)
])

model.build(input_shape=(None, 28*28))
model.summary()

opt = optimizers.Adam(lr=LR)

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = path.sep.join(['logs', current_time])
summary_writer = tf.summary.create_file_writer(log_dir)

sample_batch = next(iter(train_dataset))[0]
sample_image = sample_batch[0]
sample_image = tf.reshape(sample_image, [1, 28, 28, 1])
with summary_writer.as_default():
    tf.summary.image('Training sample: ', sample_image, step=0)

for e in range(EPOCHS):
    train_epoch_loss = 0
    train_total_num, val_total_num = 0, 0
    val_correct = 0
    for batch_x, batch_y in train_dataset:
        batch_x = tf.reshape(batch_x, [-1, 28*28])

        with tf.GradientTape() as tape:
            logits = model(batch_x)
            batch_y_onehot = tf.one_hot(batch_y, depth=10)
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

    val_images = batch_x[:25]
    val_images = tf.reshape(val_images, [-1, 28, 28, 1])

    with summary_writer.as_default():
        tf.summary.scalar('train-loss', float(train_epoch_loss/train_total_num), step=e)
        tf.summary.scalar('val-acc', float(val_correct/val_total_num), step=e)
        tf.summary.image('val-onebyone-images:', val_images, max_outputs=25, step=e)

        val_images = tf.reshape(val_images, [-1, 28, 28])
        figure = image_grid(val_images)
        tf.summary.image('val-images:', plot_to_image(figure), step=e)