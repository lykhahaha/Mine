import numpy as np
import os

class ImageNetHelper:
    def __init__(self, config):
        self.config = config

        # build label mappings and validation blacklist
        self.label_mappings = self.build_class_labels()
        self.val_blacklist = self.build_blacklist()

    def build_class_labels(self):
        # load contents of file that maps WordNet IDs to integers, then initialize label mappings dictionary
        rows = open(self.config.WORD_IDS).read().strip().split('\n')
        label_mappings = {}

        # loop over lines in WORD_IDS
        for row in rows:
            # split row into WordNet ID, label integer and human readable label
            word_id, label, hr_label = row.split()

            # update label mappings dictionary using word ID as key and label as the value
            # substract 1 from label since MATLAB is one-indexed while Python is zero-indexed
            label_mappings[word_id] = int(label) - 1

        return label_mappings
    
    def build_blacklist(self):
        rows = open(self.config.VAL_BLACKLIST).read()
        rows = set(rows.strip().split('\n'))

        return rows

    def build_training_set(self):
        # load contents of training input file that lists partial image ID and image number
        rows = open(self.config.TRAIN_LIST).read().strip().split('\n')
        paths, labels = [], []

        # loop over lines in TRAIN_LIST
        for row in rows:
            # break row into partial path and image number
            partial_path, image_num = row.split()

            # construct full path to training image, then grab word ID from path and use it to determine interget class label
            path = os.path.sep.join([self.config.IMAGES_PATH, 'train', f'{partial_path}.JPEG'])
            word_id = partial_path.split('/')[0]
            label = self.label_mappings[word_id]

            # update respective paths and label list
            paths.append(path)
            labels.append(label)

        return np.array(paths), np.array(labels)

    def build_validation_set(self):
        # initialize list of image paths and labels
        paths, labels = [], []

        # load contents of file that lists partial validation image filenames
        val_filenames = open(self.config.VAL_LIST).read().strip().split('\n')
        
        # load contents of file that contains actual ground-truth integer class labels for validation set
        val_labels = open(self.config.VAL_LABELS).read().strip().split('\n')
        
        for row, label in zip(val_filenames, val_labels):
            # break row into partial path and image number
            partial_path, image_num = row.split()

            # check to see the image is in blacklist
            if image_num not in self.val_blacklist:
                # construct full path to validation image, then update respective paths and labels lists
                path = os.path.sep.join([self.config.IMAGES_PATH, 'val', f'{partial_path}.JPEG'])
                paths.append(path)
                labels.append(int(label) - 1)

        return np.array(paths), np.array(labels)