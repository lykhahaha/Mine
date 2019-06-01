# taking input set of numpy arrays and writing them to HDF5 format
import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, output_path, data_key='images', buf_size=1000):
        # check to see if output path exists, and if so, raise an exception
        if os.path.exists(output_path):
            raise ValueError("The supplied 'output_path' already exists and cannot be overwritten. Manually delete the file before continuing.", output_path)

        # open HDF5 file for writing and create 2 datasets: one to store images/features and another to store class labels
        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset(data_key, dims, dtype='float')
        self.labels = self.db.create_dataset('labels', (dims[0],), dtype='int')

        # store buffer size, then initialize buffer itself along with index into datasets
        self.buf_size = buf_size
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0
    
    def add(self, rows, labels):
        # add rows and labels to buffer
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        # check to see if buffer need flushing to disk
        if len(self.buffer['data']) >= self.buf_size:
            self.flush()
    
    def flush(self):
        # write buffers to disk then reset buffer
        next_idx = self.idx+len(self.buffer['data'])
        self.data[self.idx:next_idx] = self.buffer['data']
        self.labels[self.idx:next_idx] = self.buffer['labels']
        self.idx = next_idx
        self.buffer = {'data': [], 'labels': []}
    
    def storeClassLabels(self, class_labels):
        # create dataset to store actual class label names, then store class labels
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset('label_names', (len(class_labels), ), dtype=dt)
        label_set[:] = class_labels
    
    def close(self):
        # check to see if there are any other entries in buffer that need flushing to disk
        if len(self.buffer['data']) > 0:
            self.flush()

        # close dataset
        self.db.close()