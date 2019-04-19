from keras.models import Model
from keras.layers import Dropout, Activation, Input, concatenate, Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.applications import VGG16
from keras import backend as K
from pyimagesearch.nn.conv import EltWiseProduct

def saliency_model(image_size=256, downsampling_factor_net=8, downsampling_factor_net=10):
    input_shape = Input(shape=(image_size, image_size, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_shape)

    for layer in base_model.layers[:14]:
        layer.trainable = False

    pretrained_weights = []
    for i in range(20, len(base_model.layers), 2):
        pretrained_weights.append([base_model.get_weights()[i], base_model.get_weights()[i+1]])

    head_model = base_model.output
    head_model = MaxPooling2D((2, 2), strides=(1, 1), padding='same')(head_model)
    head_model = Conv2D(512, (3, 3), padding='same', activation='relu', weights=pretrained_weight)(head_model)
    
    merge = concatenate([base_model.get_layer('block3_pool').output, base_model.get_layer('block4_pool').output, base_model.get_layer('block5_conv3').output], axis=-1)
    dropout = Dropout(0.5)(merge)
    init_conv = Conv2D(64, (3, 3), padding='valid', )