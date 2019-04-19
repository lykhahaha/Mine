from os import path

# define path to input images we will be using to build training crops
INPUT_IMAGES = 'ukbench100'

# define path to temp output directories
BASE_OUTPUT = 'output'
IMAGES = path.sep.join([BASE_OUTPUT, 'images'])
LABELS = path.sep.join([BASE_OUTPUT, 'labels'])

# define path to HDF5 files
INPUT_DB = path.sep.join([BASE_OUTPUT, 'inputs.hdf5'])
OUTPUT_DB = path.sep.join([BASE_OUTPUT, 'outputs.hdf5'])

# define path to output model file and plot file
MODEL_PATH = path.sep.join([BASE_OUTPUT, 'srcnn.model'])
PLOT_PATH = path.sep.join([BASE_OUTPUT, 'plot.png'])

# initialize batch size and number of epochs for training
BATCH_SIZE = 128
NUM_EPOCHS = 10

# initialize scale(how large image enlarge to) along with input width and height dimensions to our SRCNN
SCALE = 2.
INPUT_DIM = 33

# label size should be output spatial dimensions of SRCNN while padding ensures we properly crop label ROI
LABEL_SIZE = 21
PAD = int((INPUT_DIM - LABEL_SIZE) / 2.)

# stride control step size of our sliding window
STRIDE = 14