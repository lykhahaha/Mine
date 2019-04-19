from os import path

DATASET = 'ukbench100'

OUTPUT_BASE = 'output'
IMAGES = path.sep.join([OUTPUT_BASE, 'images'])
LABELS = path.sep.join([OUTPUT_BASE, 'labels'])

INPUTS_DB = path.sep.join([OUTPUT_BASE, 'inputs.hdf5'])
OUTPUTS_DB = path.sep.join([OUTPUT_BASE, 'outputs.hdf5'])

BATCH_SIZE = 128
NUM_EPOCHS = 10

MODEL_PATH = path.sep.join([OUTPUT_BASE, 'srcnn.model'])
PLOT_PATH = path.sep.join([OUTPUT_BASE, 'plot_srcnn.png'])

INPUT_DIM = 33
LABEL_SIZE = 21
SCALE = 2.
PAD = int((INPUT_DIM-LABEL_SIZE)/2.)
STRIDE = 14