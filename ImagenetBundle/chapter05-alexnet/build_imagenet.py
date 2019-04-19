# USAGE
# python build_imagenet.py
# python ~/anaconda3/lib/python3.7/site-packages/mxnet/tools/im2rec.py ../datasets/imagenet/lists/ '' --recursive --resize 256 --encoding '.jpg' --quality 100 --num-thread 16
from config import imagenet_alexnet_config as config
from sklearn.model_selection import train_test_split
from pyimagesearch.utils import ImageNetHelper
import numpy as np
import progressbar
import json
import cv2

# initialize ImageNet helper and use it to construct set of training and testing data
print('[INFO] loading image paths...')
inh = ImageNetHelper(config)
train_paths, train_labels = inh.build_training_set()
val_paths, val_labels = inh.build_validation_set()

# partition stratified sampling from training set to construct a testing set
print('[INFO] constructing splits...')
train_paths, test_paths, train_labels, test_labels = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_IMAGES, stratify=train_labels, random_state=42)

# construct a list pairing training, validation and testing image paths along with corresponding labels and output list files
DATASETS = [
    ('train', train_paths, train_labels, config.TRAIN_MX_LIST),
    ('val', val_paths, val_labels, config.VAL_MX_LIST),
    ('test', test_paths, test_labels, config.TEST_MX_LIST)
]

# initialize R, G, B list for average
R, G, B = [], [], []

# loop over DATASETS
for d_type, paths, labels, output_path in DATASETS:
    # open output file for writing
    print(f'[INFO] building {d_type}...')
    f = open(output_path, 'w')

    # initialize progress bar
    widgets = ['Building list: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(labels), widgets=widgets).start()

    # loop over each image and label
    for i, (path, label) in enumerate(zip(paths, labels)):
        # write image index, label and output path to file
        row = '\t'.join([str(i), str(label), path])
        f.write(f'{row}\n')
        
        # if we are building training dataset, compute mean of each channel in the image
        if d_type == 'train':
            image = cv2.imread(path)
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        pbar.update(i)

    # close output file
    pbar.finish()
    f.close()

# construct a dictionary of averages, then serialize means to JSON file
print('[INFO] serializing means...')
D = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(D))
f.close()