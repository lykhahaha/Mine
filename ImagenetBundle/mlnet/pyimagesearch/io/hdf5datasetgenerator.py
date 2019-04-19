from keras.utils import to_categorical
import numpy as np
import h5py

class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size, preprocessors=None, aug=None, binarize=True, classes=2):
        # store batch size, preprocessors and data augmentation, whether or not labels should be binarized, along with total number of classes
        self.batch_size = batch_size
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes

        # open HDF5 database for reading and determine total number of entries in database
        self.db = h5py.File(db_path, 'r')
        self.num_images = self.db['labels'].shape[0]

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        def preprocessing(self, images, labels):
            # check to see if labels should be binarized
                if self.binarize:
                    labels = to_categorical(labels, self.classes)

                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize list of processed images
                    proc_images = []

                    # loop over the images
                    for image in images:
                        # loop over the preprocessors and apply it
                        for p in self.preprocessors:
                            image = p.preprocess(image)

                        # update list of processed images
                        proc_images.append(image)
                    
                    images = np.array(proc_images)

                # check to see data augmentation and apply it
                if self.aug is not None:
                    images, labels = next(self.aug.flow(images, labels, batch_size=self.batch_size))

                return images, labels

        # keep looping infinitely -- model will stop once we have reach the desired number of epochs
        while epochs < passes:
            # loop over HDF5 dataset
            for i in range(0, self.num_images, self.batch_size):
                # extract images and labels from HDF5 dataset
                images = self.db['images'][i:i+self.batch_size]
                labels = self.db['labels'][i:i+self.batch_size]
                
                yield preprocessing(self, images, labels)

            if self.num_images % self.batch_size == 0:
                remainder = self.batch_size * (self.num_images//self.batch_size)
                images = self.db['images'][remainder:]
                labels = self.db['labels'][remainder:]

                yield preprocessing(self, images, labels)

            # increment total number of epochs
            epochs += 1

    def close(self):
        # close database
        self.db.close()