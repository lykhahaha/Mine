from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1

        # if we are using "channels first", update the input shape
        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1

        # Layer Type    Output Size     Filter Size     Stride
        # INPUT IMAGE   32 × 32 × 3
        # CONV          32 × 32 × 32      3 × 3         K = 32
        # ACT           32 × 32 × 32
        # BN            32 × 32 × 32
        # CONV          32 × 32 × 32      3 × 3         K = 32
        # ACT           32 × 32 × 32
        # BN            32 × 32 × 32
        # POOL          16 × 16 × 32      2 × 2

        # DROPOUT       16 × 16 × 32

        # CONV          16 × 16 × 64      3 × 3         K = 64
        # ACT           16 × 16 × 64
        # BN            16 × 16 × 64
        # CONV          16 × 16 × 64      3 × 3         K = 64
        # ACT           16 × 16 × 64
        # BN            16 × 16 × 64
        # POOL          8 × 8 × 64        2 × 2

        # DROPOUT       8 × 8 × 64

        # FC            512
        # ACT           512
        # BN            512

        # DROPOUT       512

        # FC            10
        # SOFTMAX       10

        # first set CONV => ReLu => BN => CONV => ReLU => BN => POOL layer
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # apply dropout for this layer
        model.add(Dropout(0.25))

        # second set CONV => ReLu => BN => CONV => ReLU => BN => POOL layer
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # apply dropout for this layer
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, activation='softmax'))

        return model