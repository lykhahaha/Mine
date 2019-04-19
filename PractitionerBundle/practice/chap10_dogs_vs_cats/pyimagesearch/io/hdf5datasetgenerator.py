import h5py
import numpy as np
from keras.utils import to_categorical

class HDF5DatasetGenerator:
    def __init__(self, db_path, preprocessors=None, batch_size=32, aug=None, binarize=True, classes=2):
        self.db = h5py.File(db_path, 'r')
        self.num_images = len(self.db['labels'])
        self.preprocessors = preprocessors
        self.batch_size = batch_size
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

    def generator(self, passes=np.inf):
        epochs = 0
        while epochs < passes:
            for i in range(0, self.num_images, self.batch_size):
                images = self.db['images'][i:i+self.batch_size]
                labels = self.db['labels'][i:i+self.batch_size]

                if self.preprocessors is not None:
                    proc_images = []
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        proc_images.append(image)
                    proc_images = np.array(proc_images)

                if self.binarize:
                    labels = to_categorical(labels, num_classes=self.classes)

                if self.aug is not None:
                    images, labels = next(self.aug.flow(images, labels, self.batch_size))
                
                yield images, labels
            epochs += 1

    def close(self):
        self.db.close()