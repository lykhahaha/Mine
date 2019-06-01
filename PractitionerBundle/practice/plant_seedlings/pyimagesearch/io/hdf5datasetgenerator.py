import cv2
import h5py
import numpy as np
from keras.utils import to_categorical

class HDF5DatasetGenerator:
    def __init__(self, hdf5_path, preprocessors, aug, binarize=True, batch_size=32, num_classes=2):
        self.db = h5py.File(hdf5_path, mode='r')
        self.num_images = len(self.db['labels'])
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.batch_size = batch_size
        self.num_classes = num_classes

    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:
            for i in range(0, self.num_images, self.batch_size):
                batch_images = self.db['images'][i:i+self.batch_size]
                batch_labels = self.db['labels'][i:i+self.batch_size]

                if self.binarize:
                    batch_labels = to_categorical(batch_labels, num_classes=self.num_classes)

                if self.preprocessors:
                    proc_images = []
                    for image in batch_images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                    proc_images.append(image)
                batch_images = np.array(proc_images)

                if self.aug:
                    batch_images, batch_labels = next(self.aug.flow(batch_images, batch_labels, batch_size=self.batch_size))

                yield batch_images, batch_labels

            epochs += 1
    
    def close(self):
        self.db.close()