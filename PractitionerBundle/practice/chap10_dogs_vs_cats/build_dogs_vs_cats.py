from config import dogs_vs_cats_config as config
from pyimagesearch.io import HDF5DatasetWriter
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import progressbar
import json
import cv2
import os

# grab the list of images path, extract labels from them
train_paths = list(paths.list_images(config.IMAGES_PATH))
train_labels = [image_path.split(os.path.sep)[-1].split('.')[0] for image_path in train_paths]

# convert labels to vector
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)

# partition data using stratifying
train_paths, test_paths, train_labels, test_labels = train_test_split(train_paths, train_labels, test_size=config.NUM_TEST_IMAGES, stratify=train_labels, random_state=42)
train_paths, val_paths, train_labels, val_labels = train_test_split(train_paths, train_labels, test_size=config.NUM_VAL_IMAGES, stratify=train_labels, random_state=42)

# initialize DATASETS for easily accessing
DATASETS = [
    ('train', train_paths, train_labels, config.TRAIN_HDF5),
    ('val', val_paths, val_labels, config.VAL_HDF5),
    ('test', test_paths, test_labels, config.TEST_HDF5)
]

# initialize preprocessor
aap = AspectAwarePreprocessor(256, 256)

# construct list of R, G, B for mean 
R, G, B = [], [], []

# loop over DATASETS
for d_type, image_paths, labels, output_path in DATASETS:
    #initialize HDF5 dataset writer
    writer = HDF5DatasetWriter(output_path, (len(labels), 256, 256, 3))

    # construct progressbar
    widgets = [f'Building {d_type}: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(labels), widgets=widgets).start()

    # loop over image path
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        image = cv2.imread(image_path)
        image = aap.preprocess(image)

        if d_type == 'train':
            b, g, r = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        
        writer.add([image], [label])
        pbar.update(i)
    
    writer.close()
    pbar.finish()

# save mean of R, G, B to json file
print('[INFO] serializing means...')
dataset_mean = {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f = open(config.DATASET_MEAN, 'w')
f.write(json.dumps(dataset_mean))
f.close()