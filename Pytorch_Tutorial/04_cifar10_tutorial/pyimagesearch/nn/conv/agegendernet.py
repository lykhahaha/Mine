from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras.models import Sequential

class AgeGenderNet:
    @staticmethod
    def build(width, height, depth, classes, reg=None):
        # define input shape chan batch channel dimension
        input_shape = (height, width, depth)
        chan_dim = -1

        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)
            chan_dim = 1

        model = Sequential()
        # Block #1: first Conv -> ReLU, Pool layer set
        model.add(Conv2D(96, (7, 7), strides=(4, 4), activation='relu', kernel_regularizer=l2(reg), input_shape=input_shape))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(Dropout(0.25))

        # Block #2: second Conv -> ReLU -> Pool layer set
        model.add(Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(Dropout(0.25))

        # Block #3: second Conv -> ReLU -> Pool layer set
        model.add(Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
        model.add(Dropout(0.25))

        # Block #4: first set of FC -> ReLU layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Dropout(0.5))

        # Block #5: secondset of FC -> ReLU layers
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, activation='softmax'))

        return model