from keras.applications import VGG16
import argparse

# construct argument parser and parse the argument
ap =  argparse.ArgumentParser()
ap.add_argument('-i', '--include-top', type=int, default=1, help='whether or not to include top of CNN')
args = vars(ap.parse_args())

# load VGG16
print('[INFO] loading VGG16 model...')
model = VGG16(weights='imagenet', include_top=args['include_top']>0)
print('[INFO] showing layers...')

# loop over the model to show layers
for i, layer in enumerate(model.layers):
    print(f'[INFO] {i}\t{layer.__class__.__name__}')