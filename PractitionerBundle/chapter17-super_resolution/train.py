import matplotlib
matplotlib.use('Agg')

from config import sr_config as config
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.nn.conv import SRCNN
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

def super_res_generator(input_datagen, target_datagen):
    # start infinite loop for training data
    while True:
        # grab next input images and target outputs, discarding class labels
        input_data, target_data = next(input_datagen)[0], next(target_datagen)[0]

        yield input_data, target_data

# initialize input images and target output images generators
iap = ImageToArrayPreprocessor()
inputs = HDF5DatasetGenerator(config.INPUT_DB, config.BATCH_SIZE, preprocessors=[iap])
targets = HDF5DatasetGenerator(config.OUTPUT_DB, config.BATCH_SIZE, preprocessors=[iap])

# initialize model and optimizer
print('[INFO] compiling model...')
opt = Adam(decay=1e-3/config.NUM_EPOCHS)
model = SRCNN.build(width=config.INPUT_DIM, height=config.INPUT_DIM, depth=3)
model.compile(opt, loss='mse')

# train model using generator
H = model.fit_generator(super_res_generator(inputs.generator(), targets.generator()), epochs=config.NUM_EPOCHS, verbose=2, steps_per_epoch=inputs.num_images//config.BATCH_SIZE)

# save model to file
print('[INFO] serializing model...')
model.save(config.MODEL_PATH)

# close dataset
inputs.close()
targets.close()

# plot training loss
plt.style.use('ggplot')
plt.figure()
plt.plot(H.history['loss'], label='loss')
plt.title('Loss on super resolution training')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
plt.savefig(config.PLOT_PATH)