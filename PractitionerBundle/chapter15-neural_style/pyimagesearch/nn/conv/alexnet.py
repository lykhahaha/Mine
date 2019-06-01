from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.regularizers import l2
from keras import backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        # initialize the model
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1

        # if using channel_first, update input shape
        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1

        # Layer Type    Output Size     Filter Size     Stride      # filters
        # INPUT IMAGE   227 × 227 × 3
        
        # CONV          55 × 55 × 96    11 × 11         4 × 4       K = 96
        # ACT           55 × 55 × 96
        # BN            55 × 55 × 96
        # POOL          27 × 17 × 96     3 × 3          2 × 2
        # DROPOUT       27 × 27 × 96

        # CONV          27 × 27 × 256    5 × 5                      K = 256
        # ACT           27 × 27 × 256
        # BN            27 × 27 × 256
        # POOL          13 × 13 × 256     3 × 3         2 × 2
        # DROPOUT       13 × 13 × 256

        # CONV          13 × 13 × 384     3 × 3                     K = 384
        # ACT           13 × 13 × 384
        # BN            13 × 13 × 384
        # CONV          13 × 13 × 384     3 × 3                     K = 384
        # ACT           13 × 13 × 384
        # BN            13 × 13 × 384

        # CONV          13 × 13 × 256     3 × 3                     K = 256
        # ACT           13 × 13 × 256
        # BN            13 × 13 × 256

        # POOL          13 × 13 × 256     3 × 3         2 × 2
        # DROPOUT       6 × 6 × 256

        # FC            4096
        # ACT           4096
        # BN            4096

        # DROPOUT       4096

        # FC            4096
        # ACT           4096
        # BN            4096

        # DROPOUT       4096

        # FC            1000
        # SOFTMAX       1000

        # Block #1: first CONV -> ReLU -> POOL layer set
        model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', kernel_regularizer=l2(reg), input_shape=input_shape))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV -> ReLU -> POOL layer set
        model.add(Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: CONV -> ReLU -> CONV -> ReLU -> CONV -> ReLU
        model.add(Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first set of FC -> ReLU layers
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Dropout(0.5))

        # Block #5: first set of FC -> ReLU layers
        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, activation='softmax', kernel_regularizer=l2(reg)))

        return model