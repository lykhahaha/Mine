from config import dogs_and_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, MeanPreprocessor, CropPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from keras.models import load_model
import progressbar
import json
import numpy as np
import cv2
import argparse
import pandas as pd

# construct argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--submit', required=True, help='path to submission file')
args = vars(ap.parse_args())

# load RGB means for json
means = json.loads(open(config.DATASET_MEAN).read())

# initialize image preprocessors
mp, cp, iap = MeanPreprocessor(means['R'], means['G'], means['B']), CropPreprocessor(227, 227), ImageToArrayPreprocessor()

# load model
print('[INFO] loading model...')
model = load_model(config.MODEL_PATH)

# initialize dataset generator
test_gen = HDF5DatasetGenerator(config.PUBLIC_TEST_HDF5, batch_size=64, preprocessors=[mp])
preds = []

# initialize progressbar
widgets = ['Evaluating: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=test_gen.num_images//64, widgets=widgets)

# loop over single pass of test data
for i, (images, labels) in enumerate(test_gen.generator(passes=1)):
    # loop over individual images
    for image in images:
        # apply crop preprocessor
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(crop) for crop in crops], dtype='float32')

        # predict on the crops
        pred = model.predict(crops)
        preds.append(pred.mean(axis=0))
    pbar.update(i)

pbar.finish()

# read sample submission
df = pd.DataFrame({
    'id': np.array(range(1, test_gen.num_images+1)),
    'label': np.array(preds).argmax(axis=1)
})
df.to_csv(args['submit'])

# close database
test_gen.close()