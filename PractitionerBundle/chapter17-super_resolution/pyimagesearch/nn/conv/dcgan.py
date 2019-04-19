from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2DTranspose, Conv2D, Flatten, Dense, Reshape, LeakyReLU
from keras.utils import plot_model

class DCGAN:
    @staticmethod
    def build_generator(dim, depth, channels=1, input_dim=100, output_dim=512):
        # initialize model along with input shape to be channels last and channels dimension itself
        model = Sequential()
        input_shape = (dim, dim, depth)
        chan_dim = -1

        # first set FC -> ReLU -> BN layers
        model.add(Dense(output_dim, activation='relu', input_dim=input_dim))
        model.add(BatchNormalization())

        # second set of FC -> ReLU -> BN layers, this time preparing number of FC nodes to be reshaped into a volume
        model.add(Dense(dim * dim * depth, activation='relu'))
        model.add(BatchNormalization())

        # reshape output of previous layer set, upsample + apply transposed convolution, ReLU then BN
        model.add(Reshape(input_shape))
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))

        # apply another upsample and transposed convolution, but this time TANK activation
        model.add(Conv2DTranspose(channels, (5, 5), strides=(2, 2), padding='same', activation='tanh'))

        return model

    @staticmethod
    def build_discriminator(width, height, depth, alpha=0.2):
        # initialize model along with input shape to be channels last
        model = Sequential()
        input_shape = (height, width, depth)

        # first set of Conv -> ReLU layers
        model.add(Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=input_shape))
        model.add(LeakyReLU(alpha=alpha))

        # second set of Conv -> ReLU layers
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=alpha))

        # first set of FC -> ReLU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=alpha))

        # sigmoid layer outputting a single value
        model.add(Dense(1, activation='sigmoid'))

        return model