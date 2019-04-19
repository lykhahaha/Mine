from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        input_shape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)

        # Layer Type    Output Size    Filter Size     Stride
        # INPUT IMAGE   28 × 28 × 1
        # CONV          28 × 28 × 20    5 × 5           K = 20
        # ACT           28 × 28 × 20
        # POOL          14 × 14 × 20    2 × 2
        # CONV          14 × 14 × 50    5 × 5           K = 50
        # ACT           14 × 14 × 50
        # POOL          7 × 7 × 50      2 × 2
        # FC            500
        # ACT           500
        # FC            10
        # SOFTMAX       10
        
        # first set CONV => ReLU => POOL layers
        model.add(Conv2D(20, (5, 5), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set CONV => ReLU => POOL layers
        model.add(Conv2D(50, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => ReLU layers
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))

        # softmax classifier
        model.add(Dense(classes, activation='softmax'))

        return model
