from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.regularizers import l2
from keras import backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        # config channels last or channels first
        input_shape = (width, height, depth)
        chan_dims = -1

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, width, height)
            chan_dims = 1

        model = Sequential()

        # Block #1: CONV -> ReLU -> POOL layer
        model.add(Conv2D(96, (11, 11), (4, 4), padding='same', activation='relu', kernel_regularizer=l2(reg), input_shape=input_shape))
        model.add(BatchNormalization(axis=chan_dims))
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: CONV -> ReLU -> POOL layer
        model.add(Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dims))
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: CONV -> ReLU -> CONV -> ReLU -> CONV -> ReLU
        model.add(Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dims))
        model.add(Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dims))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dims))
        model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first set of FC -> ReLU layer
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dims))
        model.add(Dropout(0.5))

        # Block #5: second set of FC -> ReLU layer
        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dims))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(1000, activation='softmax', kernel_regularizer=l2(reg)))

        return model