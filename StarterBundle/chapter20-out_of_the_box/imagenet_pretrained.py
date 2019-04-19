# USAGE
# python imagenet_pretrained.py --image example_images/example_01.jpg --model vgg19
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TENSORFLOW only
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

# construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-m', '--model', type=str, default='vgg16', help='name of pre-trained network to use')
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes inside Keras
MODELS = {
    'vgg16': VGG16,
    'vgg19': VGG19,
    'inception': InceptionV3,
    'xception': Xception, # TENSORFLOW only
    'resnet': ResNet50
}

# ensure a valid model name was supplied via cmd argument
if args['model'] not in MODELS.keys():
    raise AssertionError('The --model command line argument should be a key in the MODELS dictionary')

# initialize the input image shape (224x224 pixels) along with the pre-process function
input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# if using InceptionV3 or Xception network, need to set input_shape (299, 299)
if args['model'] in ('inception', 'xception'):
    input_shape = (299, 299)
    preprocess = preprocess_input

# load network weights from disk
print(f"[INFO] loading {args['model']}...")
Network = MODELS[args['model']]
model = Network(weights='imagenet')

# load input image using Keras helper utility and make sure image is resized to input_shape
print(f"[INFO] loading and preprocess image...")
image = load_img(args['image'], target_size=input_shape)
image = img_to_array(image)

# put batch to image size to make it from (input_shape[0], input_shape[1], 3) to (1, input_shape[0], input_shape[1], 3)
image = np.expand_dims(image, axis=0)

# preprocess the image using proper preprocess function based on the pretrained model loaded
image = preprocess(image) # 82.  81.  82.  78.  --->  -0.35686272 -0.36470586 -0.35686272 -0.38823527

# classify the image
print(f"[INFO] classifying image with {args['model']}...")
preds = model.predict(image) # [[4.01924121e-15 2.49983529e-14 2.33872844e-13 5.91792579e-14 8.83753452e-14 1.90098832e-13 1.89975141e-14 7.52140524e-14 1.06349588e-13 3.62107601e-14 1.12857509e-14 1.10510110e-14]]
P = imagenet_utils.decode_predictions(preds) # [[('n03982430', 'pool_table', 0.9999995), ('n03942813', 'ping-pong_ball', 3.5357147e-07), ('n03452741', 'grand_piano', 1.6495245e-07), ('n04081281', 'restaurant', 1.2804391e-08), ('n02777292', 'balance_beam', 4.314796e-09)]]

# loop over predictions and display top-5 + probabilities to terminal
for i, (imagenetID, label, prob) in enumerate(P[0]):
    print(f'{i+1}. {label}: {prob*100:.2f}%')

# load image via OpenCV, draw top predictions on the image and display image to our screen
orig = cv2.imread(args['image'])
imagenetID, label, prob = P[0][0]
cv2.putText(orig, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow('Classification', orig)
cv2.waitKey(0)