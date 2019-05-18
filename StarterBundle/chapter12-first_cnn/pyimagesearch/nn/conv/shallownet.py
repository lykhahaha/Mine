from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be "channels last"
        model = Sequential()
        input_shape = (height, width, depth)

        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)

        # define the first (and only) CONV => ReLU layer
        model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=input_shape))
        # softmax classifier
        model.add(Flatten())
        model.add(Dense(classes, activation='softmax'))

        # return the constructed network architecture
        return model