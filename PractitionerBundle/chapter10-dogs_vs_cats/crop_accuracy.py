from config import dogs_and_cats_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, SimplePreprocessor, MeanPreprocessor, CropPreprocessor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.utils.ranked import rank5_accuracy
from keras.models import load_model
import numpy as np
import progressbar
import json

# load RGB means from json
means = json.loads(open(config.DATASET_MEAN).read())

# initialize image preprocessors
sp, mp, cp, iap = SimplePreprocessor(227, 227), MeanPreprocessor(means['R'], means['G'], means['B']), CropPreprocessor(227, 227), ImageToArrayPreprocessor()

# load model
print('[INFO] loading model...')
model = load_model(config.MODEL_PATH)

# initialize testing dataset generator, then make predictions on testing data
print('[INFO] predicting in testing data (no crops)...')
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, batch_size=64, preprocessors=[sp, mp, iap])
preds = model.predict_generator(test_gen.generator(), steps=test_gen.num_images//64)

# compute rank-1 and rank-5 accuracies
rank_1, _ = rank5_accuracy(preds, test_gen.db['labels'])
print(f'[INFO] rank_1: {rank_1*100:.2f}')
test_gen.close()

# re-initialize testing set generator, this time excluding SimplePreprocessor
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, batch_size=64, preprocessors=[mp])
preds = []

# initialize the progress bar
widgets = ['Evaluating: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=test_gen.num_images//64, widgets=widgets).start()

# loop over a single pass of test data
for i, (images, labels) in enumerate(test_gen.generator(passes=1)):
    # loop over each of individual images
    for image in images:
        # apply the crop preprocessor to the image to generate 10 separate crops
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype='float32')

        # predict on the crops and then average them together
        pred = model.predict(crops)
        preds.append(pred.mean(axis=0))

    pbar.update(i)

pbar.finish()

# compute rank-1 accuracy
print('[INFO] predicting in testing data (with crops)...')
rank_1, _ = rank5_accuracy(preds, test_gen.db['labels'])
print(f'[INFO] rank_1: {rank_1*100:.2f}')
test_gen.close()