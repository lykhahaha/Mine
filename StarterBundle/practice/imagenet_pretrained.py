from keras.applications import VGG16, VGG19, ResNet50, InceptionV3, Xception, imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import argparse
import cv2

# Construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-model', '--model', type=str, default='vgg16', help='name of pre-trained weights model to use')
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = vars(ap.parse_args())

# construct MODELS
MODELS = {
    'vgg16': VGG16,
    'vgg19': VGG19,
    'resnet': ResNet50,
    'inception': InceptionV3,
    'xception': Xception
}

if args['model'] not in MODELS.keys():
    raise AssertionError('The model is not supported')

# preprocess method based on the model
preprocess = imagenet_utils.preprocess_input
input_shape = (224, 224)

if args['model'] in ('inception', 'xception'):
    preprocess = preprocess_input
    input_shape = (299, 299)

# load image
print('[INFO] loading test images...')
image = load_img(args['image'], target_size=input_shape)
image = img_to_array(image)

# add new axis to image to feed into network
image = np.expand_dims(image, axis=0)

# preprocess image
image = preprocess(image)

# predict label
print('[INFO] predicting label...')
Network = MODELS[args['model']]
model = Network(weights='imagenet')
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

for i, (imagenet_id, label, prob) in enumerate(P[0]):
    print(f'{i+1}. {label}: {prob*100:.2f}')

# display image and its top-1 prediction
imagenet_id, label, prob = P[0][0]
orig = cv2.imread(args['image'])
cv2.putText(orig, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imwrite(f"{args['image']}_{args['model']}.png", orig)