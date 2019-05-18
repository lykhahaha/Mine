import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2

class VGG16(Model):
    def __init__(self, input_shape, wd=None, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        model = Sequential()

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd), input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(wd)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))

        self.model = model

    def call(self, x):
        x = self.model(x)

        return x