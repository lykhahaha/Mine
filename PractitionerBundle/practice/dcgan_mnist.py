from pyimagesearch.nn.conv import DCGAN
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Input, Model
from sklearn.utils import shuffle
import os
import cv2
from imutils import build_montages
import numpy as np
import argparse

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='path to output directory')
ap.add_argument('-e', '--num-epochs', type=int, default=50, help='# of epochs to train')
ap.add_argument('-b', '--batch-size', type=int, default=128, help='size of batch')
args = vars(ap.parse_args())

# define epoch # and batch size
NUM_EPOCHS = args['num_epochs']
BATCH_SIZE = args['batch_size']

# load dataset and scale it to range [-1, 1]
(trainX, _), (testX, _) = mnist.load_data()
train_images = np.concatenate([trainX, testX], axis=0)
train_images = np.expand_dims(train_images, axis=-1)
train_images = (train_images.astype('float') - 127.5) / 127.5

# GAN process
disc = DCGAN.build_discriminator(28, 28, 1)
disc_opt = Adam(lr=2e-4, beta_1=0.5, decay=2e-4/NUM_EPOCHS)
disc.compile(optimizer=disc_opt, loss='binary_crossentropy')

disc.trainable=False
gen = DCGAN.build_generator(7, 64)
gan_input = Input(shape=(100,))
gan_output = disc(gen(gan_input))
gan = Model(gan_input, gan_output)
gan_opt = Adam(lr=2e-4, beta_1=0.5, decay=2e-4/NUM_EPOCHS)
gan.compile(optimizer=gan_opt, loss='binary_crossentropy')

# training
print('[INFO] training GAN...')
for epoch in range(NUM_EPOCHS):
    batches_per_epoch = int(len(train_images)/BATCH_SIZE)
    benchmark_noise = np.random.uniform(-1, 1, size=(256, 100))

    for i in range(batches_per_epoch):
        aut_images = train_images[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

        gen_noise = gen.predict(noise, batch_size=BATCH_SIZE)
        X = np.concatenate([aut_images, gen_noise], axis=0)
        y = ([1]*BATCH_SIZE) + ([0]*BATCH_SIZE)
        X, y = shuffle(X, y)

        disc_loss = disc.train_on_batch(X, y)

        noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
        gan_loss = gan.train_on_batch(noise, [1]*BATCH_SIZE)

        if i == batches_per_epoch-1:
            p = [args['output'], f'epoch_{epoch+1:04d}_output.png']
            print(f'[INFO] Step {epoch+1}: discriminator_loss={disc_loss:.6f}, adversarial_loss={gan_loss:.6f}')

            image = gen.predict(benchmark_noise)
            image = ((image * 127.5) + 127.5).astype('uint8')
            image = np.repeat(image, 3, axis=-1)
            vis = build_montages(image, (28, 28), (16, 16))[0]

            cv2.imwrite(os.path.sep.join(p), vis)