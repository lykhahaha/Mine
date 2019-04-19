from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from scipy.optimize import fmin_l_bfgs_b
from keras import backend as K
import numpy as np
import os
import cv2

class NeuralStyle:
    def __init__(self, settings):
        self.S = settings
        w, h = load_img(self.S['input_path']).size
        self.dims = (h, w)

        self.content = self.preproces(self.S['input_path'])
        self.style = self.preproces(self.S['style_path'])
        self.output = K.placeholder((1, self.dims[0], self.dims[1], 3))
        self.content = K.constant(self.content)
        self.style = K.constant(self.style)
        self.input = K.concatenate([self.content, self.style, self.output], axis=0)

        self.model = self.S['net'](weights='imagenet', include_top=False, input_tensor=self.input)

        layer_map = {layer.name: layer.output for layer in self.model.layers}

        content_output = layer_map[self.S['content_layer']]
        content_features = content_output[0, :, :, :]
        output_features = content_output[2, :, :, :]

        content_loss = self.compute_content_loss(content_features, output_features)
        content_loss *= self.S['content_weight']

        style_loss = K.variable(0.0)
        scale = 1./len(self.S['style_layer'])
        for layer in self.S['style_layer']:
            style_output = layer_map[layer]
            style_features = style_output[1, :, :, :]
            output_features = style_output[2, :, :, :]
            style_loss += (scale * self.compute_style_loss(style_features, output_features))

        style_loss *= self.S['style_weight']

        tv_loss = self.S['tv_weight']*self.compute_tv_loss(self.output)

        total_loss = content_loss + style_loss + tv_loss

        grads = K.gradients(total_loss, self.output)
        print(grads)

        outputs = [total_loss]
        print(outputs)
        outputs += grads
        print(outputs)

        self.loss_and_grads = K.function([self.output], outputs)

    def preproces(self, image_path):
        image = load_img(image_path, target_size=self.dims)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        return image

    def deprocess(self, X):
        image = X.reshape((self.dims[0], self.dims[1], 3))
        imagenet_mean = [103.939, 116.779, 123.680]
        image[:, :, 0] += imagenet_mean[0]
        image[:, :, 1] += imagenet_mean[1]
        image[:, :, 2] += imagenet_mean[2]

        image = np.clip(image, 0, 255).astype('uint8')

        return image

    def gram_mat(self, X):
        features = K.permute_dimensions(X, (2, 0, 1))
        features = K.batch_flatten(features)
        features = K.dot(features, K.transpose(features))

        return features

    def compute_content_loss(self, content_features, output_features):
        return K.sum(K.square(content_features - output_features))

    def compute_style_loss(self, style_features, output_features):
        A = self.gram_mat(style_features)
        G = self.gram_mat(output_features)
        scale = 1./((2 * 3 * self.dims[0] * self.dims[1])**2)
        return scale * K.sum(K.square(A - G))

    def compute_tv_loss(self, X):
        h, w = self.dims
        A = K.square(X[:, :h-1, :w-1, :] - X[:, 1:, :w-1, :])
        B = K.square(X[:, :h-1, :w-1, :] - X[:, :h-1, 1:, :])
        return K.sum(K.pow(A + B, 1.25))

    def transfer(self, max_val=20):
        X = np.random.uniform(0, 255, (1, self.dims[0], self.dims[1], 3)) - 128
        for i in range(self.S['iterations']):
            print(f"[INFO] starting iteration {i+1} of {self.S['iterations']}...")
            X, loss, _ = fmin_l_bfgs_b(self.loss, X.flatten(), fprime=self.grads, maxfun=max_val)
            print(f'[INFO] end of iteration {i+1}, loss: {loss:.4e}')

            image = self.deprocess(X.copy())
            cv2.imwrite(os.path.sep.join([self.S['output_path'], f'iter_{i+1}.png']), image)

    def loss(self, X):
        X = X.reshape((1, self.dims[0], self.dims[1], 3))
        loss = self.loss_and_grads([X])[0]

        return loss

    def grads(self, X):
        X = X.reshape((1, self.dims[0], self.dims[1], 3))
        grads = self.loss_and_grads([X])[1]

        return grads.flatten().astype('float')