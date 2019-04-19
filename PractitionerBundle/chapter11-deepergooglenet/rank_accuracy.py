from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor, MeanPreprocessor, SimplePreprocessor
from pyimagesearch.utils.ranked import rank5_accuracy
from pyimagesearch.io import HDF5DatasetGenerator
from keras.models import load_model
import json

# load RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize image preprocessors
sp, mp, iap = SimplePreprocessor(64, 64), MeanPreprocessor(means['R'], means['G'], means['G']), ImageToArrayPreprocessor()

# initialize testing dataset generator
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, 64, preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

# load pre-trained network
print('[INFO] loading network...')
model = load_model(config.MODEL_PATH)

print('[INFO] predicting on test data...')
preds = model.predict_generator(test_gen.generator(), steps=test_gen.num_images//64)

# compute rank-1 and rank-5 accuracies
rank_1, rank_5 = rank5_accuracy(preds, test_gen.db['labels'])
print(f'[INFO] rank-1: {rank_1*100:.2f}')
print(f'[INFO] rank-5: {rank_5*100:.2f}')

# close dataset
test_gen.close()