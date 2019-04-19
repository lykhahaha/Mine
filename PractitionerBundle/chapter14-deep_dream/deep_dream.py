from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K
from scipy import ndimage
import numpy as np
import cv2
import argparse

def preprocess(p):
    # load input image, convert it to Keras-compatible array, expand dimensions so we can pass it through model and preprocess it with inception
    image = load_img(p)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image

def deprocess(image):
    # we are using 'channels last' so make sure RGB channels are last dimension
    image = image.reshape((image.shape[1], image.shape[2], 3))

    # undo preprocessing done for Inception to bring image back to [0, 255]
    image /= 2.0
    image += 0.5
    image *= 255.0

    image = np.clip(image, 0, 255).astype('uint8')

    # we have been precessing images in RGB order; however, openCV assumes images are in BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def resize_image(image, size):
    resized = image.copy()
    resized = ndimage.zoom(resized, (1, float(size[0]) / resized.shape[1], float(size[1]) / resized.shape[2], 1), order=1)
    return resized

def eval_loss_and_gradients(X):
    output = fetch_loss_grads([X])
    loss, G = output[0], output[1]

    return loss, G

def gradient_ascent(X, iters, alpha, max_loss=-np.inf):
    # loop over number of iterations
    for i in range(iters):
        # compute loss and gradient
        loss, G = eval_loss_and_gradients(X)

        # if loss > max_loss, breask
        if loss > max_loss:
            break
        
        # take a step
        if (i+1)%5==0:
            print(f'[INFO] loss at {i+1}: {loss}')
        X += alpha * G

    return X

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-o', '--output', required=True, help='path to output dreamed image')
args = vars(ap.parse_args())

# define dictionary that includes layers going to use for dream and their respective weights
LAYERS = {
    'mixed2': 2.0,
    'mixed3': 0.5
}

# define number of octaves, octave scale, alpha (for gradient descent), # of iterations and max loss - tweaking these values will produce different dreams
NUM_OCTAVE = 3
OCTAVE_SCALE = 1.4
ALPHA = 1e-3
NUM_ITERS = 50
MAX_LOSS = 10.0

# indicate that Keras weights of any layer should not be updated during deep dream
K.set_learning_phase(0)

# load Inception model from disk, then grab input tensor of the model
print('[INFO] loading inception network...')
model = InceptionV3(weights='imagenet', include_top=False)
dream = model.input

# define loss value, then build a dictionary that maps name of each layer inside of Inception to actual layer object
loss = K.variable(0.0)
layer_map = {layer.name: layer for layer in model.layers}

# loop over layers that will be utilized in dream
for layer_name in LAYERS:
    # grab the output of layer we will use for dreaming, the add L2-norm of the features to the layer to loss
    x = layer_map[layer_name].output
    coeff = LAYERS[layer_name]
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    loss += coeff * K.sum(K.square(x[:, 2:-2, 2:-2, :])) / scaling

# compute gradients of dream wrt loss and then normalize
grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

# we now need to define a function that can retrieve value of loss and gradients given an input image
outputs = [loss, grads]
fetch_loss_grads = K.function([dream], outputs)

# load and preprocess input image, then grab input height and width
image = preprocess(args['image'])
dims = image.shape[1:3]

# list to store spatial dimensions that we will be resizing input image to
octave_dims = [dims]

# loop over number of octaves
for i in range(NUM_OCTAVE):
    # compute spatial dimensions for current octave, then update octave_dims
    size = [int(d / OCTAVE_SCALE**(i+1)) for d in dims]
    octave_dims.append(size)

# reverse octave_dims such that smallest dims is at the front
octave_dims = octave_dims[::-1]

# clone original image and create resized input image that matches smallest dimensions
orig = image.copy()
shrunk = resize_image(image, octave_dims[0])

# loop over octave dimensions from smallest to largest
for i, size in enumerate(octave_dims):
    # resize image nad apply gradient ascent
    print(f'[INFO] starting octave {i}...')
    image = resize_image(image, size)
    image = gradient_ascent(image, iters=NUM_ITERS, alpha=ALPHA, max_loss=MAX_LOSS)

    # to compute lost detail we need 2 images: (1) shrunk image that has been upscaled to current octave, (2) original image that has been downscaled to current octave
    upscaled = resize_image(shrunk, size)
    downscaled = resize_image(orig, size)

    # lost detail is computed via sample subtraction which we back in to the image we applied gradient ascent to
    lost = downscaled - upscaled
    image += lost

    # make original image be nwe shrunk image so we can repeat process
    shrunk = resize_image(orig, size)

# deprocess our dream and save it
image = deprocess(image)
cv2.imwrite(args['output'], image)