from .layer import EltWiseProduct
from keras.models import Model
from keras.layers import Dropout, Activation, Input, concatenate, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.applications import VGG16
from keras import backend as K
from keras import constraints, initializers, activations
import math
import tensorflow as tf
import numpy as np

class SaliencyNet:
    @staticmethod
    def build(image_rows=480, image_cols=640, downsampling_factor_net=8, downsampling_factor_product=10, reg=5e-4):
        input_shape = Input(shape=(image_rows, image_cols, 3))
        # Feature extraction network
        # vgg_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_shape)
        vgg_model = VGG16(weights=None, input_tensor=input_shape)

        # pretrained_weights = []
        # for i in range(20, len(vgg_model.get_weights()), 2):
        #     pretrained_weights.append([vgg_model.get_weights()[i], vgg_model.get_weights()[i+1]])

        base_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block4_conv3').output)

        head_model = base_model.output
        head_model = MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='block4_pool')(head_model)
        for i in range(3):
            head_model = Conv2D(512, (3, 3), padding='same', activation='relu', name=f'block5_conv{i+1}')(head_model)

        fe_model = Model(inputs=base_model.input, outputs=head_model)
        # for i, pretrained_weight in zip(range(15, 18), pretrained_weights):
        #     fe_model.layers[i].set_weights(pretrained_weight)

        merge = concatenate([fe_model.get_layer('block3_pool').output, fe_model.get_layer('block4_pool').output, fe_model.get_layer('block5_conv3').output], axis=-1)
        dropout = Dropout(0.5)(merge)
        encode_conv = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(reg))(dropout)
        encode_conv = Conv2D(1, (1, 1), padding='same', activation='relu', kernel_regularizer=l2(reg))(encode_conv)

        reg_rows = math.ceil(image_rows / downsampling_factor_net) // downsampling_factor_product
        reg_cols = math.ceil(image_cols / downsampling_factor_net) // downsampling_factor_product
        elt_prod = EltWiseProduct(kernel_initializer='zero', kernel_regularizer=l2(1/(reg_rows*reg_cols)))(encode_conv)
        # added = Lambda(lambda x: x * K.resize_images(x.get_shape().as_list()[1:3]))(encode_conv)
        # Ko the: added = Add()([tf.constant(1, dtype=tf.float32, shape=(32,)), encode_conv])

        # prior learning
        prior_learning = Activation('relu')(elt_prod)

        model = Model(inputs=fe_model.input, outputs=prior_learning)
        
        return model