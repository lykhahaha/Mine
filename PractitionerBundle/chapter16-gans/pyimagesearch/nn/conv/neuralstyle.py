from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import cv2
import os

class NeuralStyle:
    def __init__(self, settings):
        self.S = settings

        # grab dimensions of input shape
        w, h = load_img(self.S['input_path']).size
        self.dims = (h, w)

        # load content image and style image, forcing dimensions of input image
        self.content = self.preprocess(settings['input_path'])
        self.style = self.preprocess(settings['style_path'])
        self.content = K.variable(self.content) # or self.content = K.constant(self.content)
        self.style = K.variable(self.style) # self.style = K.constant(self.style)

        # allocate memory of our output image, then combine content, style and output into a single tensor so they can be fed to network
        self.output = K.placeholder((1, self.dims[0], self.dims[1], 3))
        self.input = K.concatenate([self.content, self.style, self.output], axis=0)

        # load model from disk
        print('[INFO] loading network...')
        self.model = self.S['net'](weights='imagenet', include_top=False, input_tensor=self.input)

        # build a dictionary that maps name of each layer to its output
        layer_map = {layer.name: layer.output for layer in self.model.layers}

        # extract features from content layer, then extract the activations from style image and output image
        content_output = layer_map[self.S['content_layer']]
        content_features = content_output[0, :, :, :]
        output_features = content_output[2, :, :, :]
        
        # compute feature reconstruction loss and weight it
        content_loss = self.feature_recon_loss(content_features, output_features)
        content_loss *= self.S['content_weight']

        # initialize style loss along with value used to weight each style layer
        style_loss = K.variable(0.0)
        weight = 1./len(self.S['style_layers'])

        # loop over style laers
        for layer in self.S['style_layers']:
            # grab current style layer and use it to extract style and output features
            style_output = layer_map[layer]
            style_features = style_output[1, :, :, :]
            output_features = style_output[2, :, :, :]

            # compute style reconstruction loss and weight it
            T = self.style_recon_loss(style_features, output_features)
            style_loss += (weight * T)

        # finish computing style loss, compute total variations loss, then compute total loss
        style_loss *= self.S['style_weight']
        tv_loss = self.S['tv_weight'] * self.tv_loss(self.output)
        total_loss = content_loss + style_loss + tv_loss

        # compute gradients out of output image wrt loss
        grads = K.gradients(total_loss, self.output)
        outputs = [total_loss]
        outputs += grads # equivalent to outputs.append(grads[0])

        # create Keras function that computes both loss and gradients together
        self.loss_and_grads = K.function([self.output], outputs)

    def preprocess(self, p):
        # load input image and preprocess it
        image = load_img(p, target_size=self.dims)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        return image

    def deprocess(self, image):
        # reshape image, reverse centering
        image = image.reshape((self.dims[0], self.dims[1], 3))
        imagent_mean = [103.939, 116.779, 123.680]
        image[:, :, 0] += imagent_mean[0]
        image[:, :, 1] += imagent_mean[1]
        image[:, :, 2] += imagent_mean[2]

        # clip value > 255 and convert to uint8
        image = np.clip(image, 0, 255).astype('uint8')


        return image

    def gram_mat(self, X):
        features = K.permute_dimensions(X, (2, 0, 1))
        features = K.batch_flatten(features)
        features = K.dot(features, K.transpose(features))

        return features

    def feature_recon_loss(self, content_features, output_features):
        # feature reconstruction loss is the squared error between content features and output features
        return K.sum(K.square(output_features - content_features))

    def style_recon_loss(self, style_features, output_features):
        # compute style reconstruction loss where A and G are gram matrix of style and generated image
        A = self.gram_mat(style_features)
        G = self.gram_mat(output_features)

        # compute scaling factor of style loss, then finish computing style reconstruction loss
        scale = 1./((2 * 3 * self.dims[0] * self.dims[1])**2)
        loss = scale * K.sum(K.square(G - A))

        return loss

    def tv_loss(self, X):
        # total variational loss encourages spatial smoothness in output page
        # avoid border pixels to avoid artifacts
        h, w = self.dims
        A = K.square(X[:, :h-1, :w-1, :] - X[:, 1:, :w-1, :])
        B = K.square(X[:, :h-1, :w-1, :] - X[:, :h-1, 1:, :])
        loss = K.sum(K.pow(A + B, 1.25))

        return loss

    def transfer(self, max_val=20):
        # generate a random noise image that will serve as a placeholder
        X = np.random.uniform(0, 255, (1, self.dims[0], self.dims[1], 3)) - 128
        # start looping over desired number of iterations
        for i in range(self.S['iterations']):
            # run L-BFGS over pixels in our generated image to minimize neural style loss
            print(f"[INFO] starting iteration {i+1} of {self.S['iterations']}...")
            X, loss, _ = fmin_l_bfgs_b(self.loss, X.flatten(), fprime=self.grads, maxfun=max_val)
            print(f'[INFO] end of iteration {i+1}, loss: {loss:.4e}')

            # deprocess generated image and write it to disk
            image = self.deprocess(X.copy())
            cv2.imwrite(os.path.sep.join([self.S['output_path'], f'iter_{i+1}.png']), image)

    def loss(self, X):
        # extract loss value
        X = X.reshape((1, self.dims[0], self.dims[1], 3))
        loss_value = self.loss_and_grads([X])[0]
        return loss_value

    def grads(self, X):
        # compute gradient
        X = X.reshape((1, self.dims[0], self.dims[1], 3))
        grads = self.loss_and_grads([X])[1]

        return grads.flatten().astype('float')