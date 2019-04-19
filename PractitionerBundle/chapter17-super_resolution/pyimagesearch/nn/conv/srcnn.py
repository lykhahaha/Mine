from keras.models import Sequential
from keras.layers import Conv2D
from keras import backend as K

class SRCNN:
    @staticmethod
    def build(width, height, depth):
        model = Sequential()
        input_shape = (height, width, depth)

        if K.image_data_format == 'channels_first':
            input_shape = (depth, height, width)

        # entire SRCNN architecture consists of 3 CONV -> ReLU layers with no zero-padding
        model.add(Conv2D(64, (9, 9), kernel_initializer='he_normal', activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, (1, 1), kernel_initializer='he_normal', activation='relu'))
        model.add(Conv2D(depth, (5, 5), kernel_initializer='he_normal', activation='relu'))

        return model