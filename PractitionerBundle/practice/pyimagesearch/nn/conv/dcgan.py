from keras.layers import Dense, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, Reshape, Flatten
from keras.models import Sequential

class DCGAN:
    @staticmethod
    def build_generator(dim, depth, channels=1, input_dim=100, output_dim=512):
        model = Sequential()
        input_shape = (dim, dim, depth)
        chan_dim = -1

        model.add(Dense(output_dim, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())

        model.add(Dense(dim * dim * depth, activation='relu'))
        model.add(BatchNormalization())

        model.add(Reshape(input_shape))
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', activation='tanh'))

        return model

    @staticmethod
    def build_discriminator(width, height, depth, alpha=2e-4):
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1

        model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=alpha))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(1, activation='sigmoid'))

        return model