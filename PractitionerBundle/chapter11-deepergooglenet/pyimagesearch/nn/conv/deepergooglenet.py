from keras.layers import BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, Dense, Flatten, Input, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K

class DeeperGoogLeNet:
    @staticmethod
    def conv_module(x, K, k_x, k_y, stride, chan_dim, padding='same', reg=None, name=None):
        # initialize CONV, ReLU and BN layer names
        conv_name, act_name, bn_name = None, None, None

        # if layer name was supplied, prepend it
        if name is not None:
            conv_name = name + '_conv'
            act_name = name + '_act'
            bn_name = name + '_bn'
        
        # define a CONV -> ReLU -> BN pattern
        x = Conv2D(K, (k_x, k_y), strides=stride, padding=padding, activation='relu', kernel_regularizer=l2(reg), name=conv_name)(x)
        x = BatchNormalization(axis=chan_dim, name=bn_name)(x)

        return x
    
    @staticmethod
    def inception_module(x, num_1x1, num_3x3_reduce, num_3x3, num_5x5_reduce, num_5x5, num_1x1_proj, chan_dim, stage, reg=None):
        # define first branch of Inception module which consists of 1x1 convolutions
        first = DeeperGoogLeNet.conv_module(x, num_1x1, 1, 1, (1, 1), chan_dim, reg=reg, name=stage+'_first')

        # define second branch of Inception module which consists of 1x1 and 3x3 convolutions
        second = DeeperGoogLeNet.conv_module(x, num_3x3_reduce, 1, 1, (1, 1), chan_dim, reg=reg, name=stage+'_second1')
        second = DeeperGoogLeNet.conv_module(second, num_3x3, 3, 3, (1, 1), chan_dim, reg=reg, name=stage+'_second2')

        # define thrid branch of Inception module which are our 1x1 and 5x5 convolutions
        third = DeeperGoogLeNet.conv_module(x, num_5x5_reduce, 1, 1, (1, 1), chan_dim, reg=reg, name=stage+'_third1')
        third = DeeperGoogLeNet.conv_module(third, num_5x5, 5, 5, (1, 1), chan_dim, reg=reg, name=stage+'_third2')

        # define fourth branch of Inception module which is POOL projection
        fourth = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name=stage+'_pool')(x)
        fourth = DeeperGoogLeNet.conv_module(fourth, num_1x1_proj, 1, 1, (1, 1), chan_dim, reg=reg, name=stage+'_fourth')

        # concatenate across channel dimension
        x = concatenate([first, second, third, fourth], axis=chan_dim, name=stage+'mixed')

        return x

    @staticmethod
    def build(width, height, depth, classes, reg=None):
        # initialize input shape to be channels last and channels dimension itself
        input_shape = (height, width, depth)
        chan_dim = -1

        # if using channels first, update input shape and channels dimension
        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1

        # define model input, followed by a sequence of CONV -> POOL -> (CONV*2) -> POOL layers
        inputs = Input(shape=input_shape)
        x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1), chan_dim, reg=reg, name='block1')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)
        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), chan_dim, reg=reg, name='block2')
        x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1), chan_dim, reg=reg, name='block3')
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool2')(x)

        # apply two Inception modules followed by a POOL
        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, chan_dim, "3a", reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, chan_dim, "3b", reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool3")(x)

        # apply 5 Inception modules followed by POOL
        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, chan_dim, '4a', reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, chan_dim, '4b', reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, chan_dim, '4c', reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, chan_dim, '4d', reg=reg)
        x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, chan_dim, '4e', reg=reg)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool4')(x)

        # apply a POOL layer (average) followed by dropout
        x = AveragePooling2D((4, 4), name='pool5')(x)
        x = Dropout(0.4, name='do')(x)

        # softmax classifier
        x = Flatten(name='flatten')(x)
        x = Dense(classes, activation='softmax', kernel_regularizer=l2(reg), name='labels')(x)

        # create the model
        model = Model(inputs, outputs=x, name='googlenet')

        return model