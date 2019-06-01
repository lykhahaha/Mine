import h5py
from os import path

class HDF5DatasetWriter:
    def __init__(self, data_path, feature_shape, buffer_size=1000, feature_key='images'):
        if path.exists(data_path):
            raise ValueError('Path does not exist')
        self.db = h5py.File(data_path, mode='w')
        self.data = self.db.create_dataset(feature_key, shape=feature_shape, dtype='float')
        self.labels = self.db.create_dataset('labels', shape=(feature_shape[0], ), dtype='int')
        self.buffer = {'data': [], 'labels': []}
        self.buffer_size = buffer_size
        self.idx = 0

    def add(self, images, labels):
        self.buffer['data'].extend(images)
        self.buffer['labels'].extend(labels)
        
        if len(self.buffer['labels']) > self.buffer_size:
            self.flush()

    def flush(self):
        next_idx = self.idx + len(self.buffer['labels'])
        self.data[self.idx:next_idx] = self.buffer['data']
        self.labels[self.idx:next_idx] = self.buffer['labels']
        self.buffer = {'data': [], 'labels': []}
        self.idx = next_idx
    
    def store_class_labels(self, class_labels):
        dt = h5py.special_dtype(vlen=str)
        self.class_labels = self.db.create_dataset('class_labels', shape=len(class_labels,), dtype=dt)
        self.class_labels[:] = class_labels
    
    def close(self):
        if len(self.buffer['labels']) > 0:
            self.flush()
        self.db.close()