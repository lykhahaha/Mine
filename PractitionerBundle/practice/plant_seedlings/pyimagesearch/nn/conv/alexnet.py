from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from keras.regularizers import l2
from keras import backend as K

class AlexNet:
    @staticmethod
    def build(width, height, depth, num_classes, reg=0.0002):
        input_shape = (height, width, depth)
        chan_dim = -1

        if K.image_data_format == 'channels_first':
            input_shape = (depth, width, height)
            chan_dim = 1

        model = Sequential()

        model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', kernel_regularizer=l2(reg), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(384, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(4096, activation='relu', kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax'))

        return model