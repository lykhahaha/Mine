from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, BatchNormalization, concatenate
from keras.models import Input, Model
from keras.regularizers import l2
from keras import backend as K

class DeeperGoogLeNet:
    @staticmethod
    def conv_module(x, K, k_x, k_y, stride, chan_dim, padding='same', reg=None, name=None):
        conv_name, bn_name = None, None

        if name is not None:
            conv_name = name + '_conv'
            bn_name = name + '_bn'
        
        x = Conv2D(K, (k_x, k_y), strides=stride, padding=padding, activation='relu', kernel_regularizer=l2(reg), name=conv_name)(x)
        x = BatchNormalization(axis=chan_dim, name=bn_name)(x)

        return x

    @staticmethod
    def inception_module(x, num_1x1, num_3x3_reduce, num_3x3, num_5x5_reduce, num_5x5, num_pool_proj, chan_dim, reg=None, stage=None):
        first = DeeperGoogLeNet.conv_module(x, num1, 1, 1, (1, 1), chan_dim, reg=reg, name=stage+'first')

        second = DeeperGoogLeNet.conv_module(x, num_3x3_reduce, 1, 1, (1, 1), chan_dim, reg=reg, name=stage+'_second1')
        second = DeeperGoogLeNet.conv_module(second, num_3x3_reduce, 3, 3, (1, 1), chan_dim, reg=reg, name=stage+'_second2')

        third = DeeperGoogLeNet.conv_module(x, num_5x5_reduce, 1, 1, (1, 1), chan_dim, reg=reg, name=stage+'_third1')
        third = DeeperGoogLeNet.conv_module(x, num_5x5, 5, 5, (1, 1), chan_dim, reg=reg, name=stage+'_third2')

        fourth = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=stage+'_fourth1')(x)
        fourth = DeeperGoogLeNet.conv_module(fourth, num_pool_proj, 1, 1, (1, 1), chan_dim, reg=reg, name=stage+'_fourth2')

        return concatenate([first, second, third, fourth], axis=chan_dim)

    @staticmethod
    def build(width, height, depth, classes, reg=None):
        input_shape = (height, width, depth)
        chan_dim = -1

        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1
        
        inputs = Input(shape=input_shape)
        x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1), chan_dim, reg=reg, name='block1')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)
        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), chan_dim, reg=reg, name='block2')
        x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1), chan_dim, reg=reg, name='block3')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool2')(x)

        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chan_dim, reg=reg, stage='3a')
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chan_dim, reg=reg, stage='3b')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool3')(x)

        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, chan_dim, reg=reg, stage='4a')
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chan_dim, reg=reg, stage='3b')
        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chan_dim, reg=reg, stage='3a')
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chan_dim, reg=reg, stage='3b')
        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chan_dim, reg=reg, stage='3a')

        