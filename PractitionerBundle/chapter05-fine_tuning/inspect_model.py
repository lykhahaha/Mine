# USAGE
# python inspect_model.py --include-top -1
from keras.applications import VGG16
import argparse

# Construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--include-top', type=int, default=1, help='whether or not to include top of CNN')
args = vars(ap.parse_args())

# load VGG16
print('[INFO] loading network...')
model = VGG16(weights='imagenet', include_top=args['include_top'] > 0)

print('[INFO] showing layers...')
for i, layer in enumerate(model.layers):
    print(f'[INFO]{i:02d}\t{layer.__class__.__name__}')

for i, layer in enumerate(model.layers):
    print(f'[INFO]{i:02d}\t{layer.name}')

for i, layer in enumerate(model.layers):
    print(i)
    for weight in layer.weights:
        print(f'{weight.shape}\t{weight.name}')
    print()