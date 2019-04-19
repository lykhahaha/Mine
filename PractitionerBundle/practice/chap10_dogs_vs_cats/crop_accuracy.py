from config import dogs_vs_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, CropPreprocessor, MeanPreprocessor, SimplePreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from keras.models import load_model
import progressbar
import numpy as np
import os
import json

# load model
model = load_model(config.MODEL_PATH)

# load means of R, G, B
means = json.loads(open(config.DATASET_MEAN).read())

# initialize preprocessors
sp, mp, cp, iap = SimplePreprocessor(227, 227), MeanPreprocessor(means['R'], means['G'], means['B']), CropPreprocessor(227, 227), ImageToArrayPreprocessor()

# initialize test generator for evaluting without cropping
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, preprocessors=[sp, mp, iap], batch_size=128)
print('[INFO] evaluating model without cropping...')
preds = model.predict_generator(test_gen.generator(), steps=test_gen.num_images//128)
rank_1, _ = rank5_accuracy(preds, test_gen.db['labels'])
print(f'[INFO] rank-1: f{rank_1*100:.2f}')

# close test_gen
test_gen.close()

# initialize test generator for evaluting with cropping
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, preprocessors=[mp], batch_size=128)
preds = []

# construct progressbar
widgets = ['Evaluating: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.Percentage()]
pbar = progressbar.ProgressBar(maxval=test_gen.num_images//128, widgets=widgets).start()

print('[INFO] evaluating model without cropping...')
for i, (images, labels) in enumerate(test_gen.generator(passes=1)):
    # loop over each image
    for image in images:
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype='float32')
        pred = model.predict(crops)
        preds.append(np.mean(pred, axis=0))

    pbar.update(i)

rank_1, _ = rank5_accuracy(preds, test_gen.db['labels'])
print(f'[INFO] rank-1: f{rank_1*100:.2f}')

# close test_gen
test_gen.close()