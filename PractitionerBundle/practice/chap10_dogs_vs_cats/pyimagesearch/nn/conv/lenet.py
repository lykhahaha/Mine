from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        input_shape = (width, height, depth)

        # first set CONV => ReLU => POOL layers
        model.add(Conv2D(20, (5, 5), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(strides=(2, 2)))

        # second set CONV => ReLU => POOL layers
        model.add(Conv2D(50, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(strides=(2, 2)))

        # first (and only) set of FC => ReLU layers
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))

        # softmax classifier
        model.add(Dense(classes, activation='softmax'))
        return model