from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        input_shape = (width, height, depth)
        chan_dim = -1

        if K.image_data_format == 'channels_first':
            input_shape = (depth, width, height)
            chan_dim = 1

        # first set Conv => ReLU => BN => Conv => ReLU => BN => POOL => Dropout layer
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))

        model.add(MaxPooling2D(strides=(2, 2)))

        model.add(Dropout(rate=0.25))

        # second set Conv => ReLU => BN => Conv => ReLU => BN => POOL layer
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))

        model.add(MaxPooling2D(strides=(2, 2)))

        model.add(Dropout(rate=0.25))

        # first set FC => ReLU => BN => Dropout
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, activation='softmax'))

        return model