from keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, concatenate, Dense, AveragePooling2D, Flatten
from keras.models import Input, Model
from keras.regularizers import l2
from keras import backend as K

class MiniGoogLeNet:
    @staticmethod
    def conv_module(x, K, k_x, k_y, stride, padding='same', chan_dim, reg=None):
        x = Conv2D(K, (k_x, k_y), strides=stride, padding=padding, activation='relu', kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chan_dim)(x)
        return x

    @staticmethod
    def inception_module(x, num_1x1, num_3x3, chan_dim, reg=None):
        first = MiniGoogLeNet.conv_module(x, num_1x1, 1, 1, (1, 1), chan_dim=chan_dim, reg=reg)
        second = MiniGoogLeNet.conv_module(x, num_3x3, 3, 3, (1, 1), chan_dim=chan_dim, reg=reg)
        return concatenate([first, second], axis=chan_dim)

    @staticmethod
    def downsample_module(x, num_3x3, chan_dim, reg=None):
        first = MiniGoogLeNet.conv_module(x, num_3x3, 3, 3, (2, 2), padding='valid', chan_dim, reg=reg)
        second = MaxPooling2D((3, 3), strides=(2, 2))(x)
        return concatenate([first, second], axis=chan_dim)

    @staticmethod
    def build(width, height, depth, classes, reg=None):
        # define input shape and set channels last
        input_shape = (width, height, depth)
        chan_dim = -1

        if K.image_data_format == 'channels_first':
            input_shape = (depth, width, height)
            chan_dim = 1
        
        # define input and first convolution
        inputs = Input(shape=input_shape)
        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1), chan_dim=chan_dim, reg=reg)

        # define inception - inception - downsample
        x = MiniGoogLeNet.inception_module(x, 32, 32, chan_dim, reg)
        x = MiniGoogLeNet.inception_module(x, 32, 48, chan_dim, reg)
        x = MiniGoogLeNet.downsample_module(x, 80, chan_dim, reg)

        # define inception - inception - inception - inception- downsample
        x = MiniGoogLeNet.inception_module(x, 112, 48, chan_dim, reg)
        x = MiniGoogLeNet.inception_module(x, 96, 64, chan_dim, reg)
        x = MiniGoogLeNet.inception_module(x, 80, 80, chan_dim, reg)
        x = MiniGoogLeNet.inception_module(x, 48, 96, chan_dim, reg)
        x = MiniGoogLeNet.downsample_module(x, 96, chan_dim, reg)

        # define inception - inception
        x = MiniGoogLeNet.inception_module(x, 176, 160, chan_dim, reg)
        x = MiniGoogLeNet.inception_module(x, 176, 160, chan_dim, reg)
        x = AveragePooling2D((7, 7))(x)
        x = Dropout(0.5)(x)

        # final layers
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', kernel_regularizer=l2(reg))(x)

        return Model(inputs, x)