from keras.layers import Layer, InputSpec
from keras import backend as K
from keras import constraints, regularizers, initializers, activations

class EltWiseProduct(Layer):
    def __init__(self, downsampling_factor=10, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.downsampling_factor = downsampling_factor
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = [InputSpec(ndim=4)]

    def build(self, input_shape):
        input_dim = [dim//self.downsampling_factor for dim in input_shape[1:3]]

        self.kernel = self.add_weight(shape=(input_dim), initializer=self.kernel_initializer, name='kernel', regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
        self.bias = None
        self.input_spec = InputSpec(ndim=4)
        self.built = True


    def call(self, x, mask=None):
        resize_image = K.resize_images(K.expand_dims(K.expand_dims(1 + self.kernel, -1), 0), self.downsampling_factor, self.downsampling_factor, data_format=K.image_data_format(), interpolation='bilinear')
        # resize_image = K.concatenate([K.zeros_like(resize_image[:, :1, :, :]), resize_image], axis=1)
        # resize_image = K.concatenate([resize_image, K.zeros_like(resize_image[:, :1, :, :])], axis=1)
        # resize_image = K.concatenate([K.zeros_like(resize_image[:, :, :1, :]), resize_image], axis=2)
        # resize_image = K.concatenate([resize_image, K.zeros_like(resize_image[:, :, :1, :])], axis=2)
        output = x * resize_image
        
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'downsampling_factor': self.downsampling_factor}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))