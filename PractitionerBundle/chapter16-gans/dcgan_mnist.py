from pyimagesearch.nn.conv import DCGAN
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.datasets import mnist
from sklearn.utils import shuffle
from imutils import build_montages
import numpy as np
import argparse
import cv2
import os

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory')
ap.add_argument('-e', '--epochs', type=int, default=50, help='# epochs to train for')
ap.add_argument('-b', '--batch-size', type=int, default=128, help='batch size for training')
args = vars(ap.parse_args())

# store epochs and batch size in convenience variables
NUM_EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']

# load MNIST dataset and stack training and testing data points so we have additional training data
print('[INFO] loading MNIST dataset...')
(trainX, _), (testX, _) = mnist.load_data()
train_images = np.concatenate([trainX, testX])

# add extra dimension for channel and scale images into range [-1, 1] (range of tanh function)
train_images = np.expand_dims(train_images, axis=-1)
train_images = (train_images.astype('float') - 127.5) / 127.5

# build generator
print('[INFO] building generator...')
gen = DCGAN.build_generator(7, 64, channels=1)

# build discriminator
print('[INFO] building discriminator...')
disc = DCGAN.build_discriminator(28, 28, 1)
disc_opt = Adam(lr=2e-4, beta_1=0.5, decay=2e-4/NUM_EPOCHS)
disc.compile(disc_opt, loss='binary_crossentropy')

# build adversarial model by first setting discriminator to not be trainable, then combine generator and discriminator
print('[INFO] building GAN...')
disc.trainable = False
gan_input = Input(shape=(100,))
gan_output = disc(gen(gan_input)) # <tf.Tensor 'sequential_2_1/dense_4/Sigmoid:0' shape=(?, 1) dtype=float32>
gan = Model(gan_input, gan_output)

# compile GAN
gan_opt = Adam(lr=2e-4, beta_1=0.5, decay=2e-4/NUM_EPOCHS)
gan.compile(gan_opt, loss='binary_crossentropy')

# randomly generate some noise
print('[INFO] start training...')
benchmark_noise = np.random.uniform(-1, 1, size=(256, 100))

# loop over epochs
for epoch in range(NUM_EPOCHS):
    # show epoch information and compute no of batches per epoch
    print(f'[INFO] starting epoch {epoch+1} of {NUM_EPOCHS}...')
    batches_per_epoch = int(len(train_images)/BATCH_SIZE)

    # loop over batches
    for i in range(batches_per_epoch):
        # initialize output path
        p = None

        # select next batch of images, then randomly generate noise for generator to predict
        image_batch = train_images[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

        # generate images using noise + generator model
        gen_images = gen.predict(noise, batch_size=BATCH_SIZE)

        # concatenate actual and generated images, construct class labels for discriminator, shuffle data
        X = np.concatenate((image_batch, gen_images))
        y = ([1]*BATCH_SIZE) + ([0]*BATCH_SIZE)
        X, y = shuffle(X, y)

        # train discriminator on data
        disc_loss = disc.train_on_batch(X, y)

        # train generator via adversarial model by random noise and training generator with discriminator weight frozen
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
        gan_loss = gan.train_on_batch(noise, [1]*BATCH_SIZE)

        # check to see if this is end of epoch, if so, initialize output path
        if i == batches_per_epoch-1:
            p = [args['output'], f'epoch_{str(epoch+1).zfill(4)}_output.png']

        # otherwise, check to see if we should visualize current batch for epoch
        else:
            # create more visualizations early in the training process
            if epoch < 10 and i%25 == 0:
                p = [args['output'], f'epoch_{str(epoch+1).zfill(4)}_step_{str(i).zfill(5)}.png']
            # visualizations later in the training process are less interesting
            elif epoch >= 10 and i%100 == 0:
                p = [args['output'], f'epoch_{str(epoch+1).zfill(4)}_step_{str(i).zfill(5)}.png']

        # check to see if we should visualize output of generator model on benchmark noise
        if p is not None:
            # shoe loss information
            print(f'[INFO] Step {epoch+1}_{i}: discriminator_loss={disc_loss:.6f}, adversarial_loss={gan_loss:.6f}')

            # make predictions on benmark noise, scale it back to range [0, 255] and generate montage
            images = gen.predict(benchmark_noise)
            images = ((images * 127.5) + 127.5).astype('uint8')
            images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28, 28), (16, 16))[0]

            # write visualization to disk
            cv2.imwrite(os.path.sep.join(p), vis)