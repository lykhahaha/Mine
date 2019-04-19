from .age_gender_config import OUTPUT_BASE, BASE_PATH, NUM_FOLD_PATHS, DATASET
from os import path

# define path to dlib facial landmark predictor
DLIB_LANDMARK_PATH = 'shape_predictor_68_face_landmarks.dat'

# define path to age/gender network + suporting files
if DATASET == 'ADIENCE':
    AGE_NETWORK_PATHS, AGE_HDF5S, AGE_LABEL_ENCODERS, AGE_MEANS = [], [], [], []
    GENDER_NETWORK_PATHS, GENDER_HDF5S, GENDER_LABEL_ENCODERS, GENDER_MEANS = [], [], [], []
    for i in range(NUM_FOLD_PATHS):
        AGE_NETWORK_PATHS.append(path.sep.join([OUTPUT_BASE, f'age_best_weights_{i}.hdf5']))
        AGE_HDF5S.append(path.sep.join([BASE_PATH, 'hdf5', f'age_test_{i}.hdf5']))
        AGE_LABEL_ENCODERS.append(path.sep.join([OUTPUT_BASE, f'age_le_{i}.cpickle']))
        AGE_MEANS.append(path.sep.join([OUTPUT_BASE, f'age_mean_{i}.json']))

        GENDER_NETWORK_PATHS.append(path.sep.join([OUTPUT_BASE, f'gender_best_weights_{i}.hdf5']))
        GENDER_HDF5S.append(path.sep.join([BASE_PATH, 'hdf5', f'gender_test_{i}.hdf5']))
        GENDER_LABEL_ENCODERS.append(path.sep.join([OUTPUT_BASE, f'gender_le_{i}.cpickle']))
        GENDER_MEANS.append(path.sep.join([OUTPUT_BASE, f'gender_mean_{i}.json']))

elif DATASET == 'IOG':
    AGE_NETWORK_PATH = path.sep.join([OUTPUT_BASE, 'age_best_weights.hdf5'])
    GENDER_NETWORK_PATH = path.sep.join([OUTPUT_BASE, 'gender_best_weights.hdf5'])

    AGE_LABEL_ENCODER = path.sep.join([OUTPUT_BASE, 'age_le.cpickle'])
    GENDER_LABEL_ENCODER = path.sep.join([OUTPUT_BASE, 'gender_le.cpickle'])

    AGE_MEAN = path.sep.join([OUTPUT_BASE, 'age_mean.json'])
    GENDER_MEAN = path.sep.join([OUTPUT_BASE, 'gender_mean.json'])
    