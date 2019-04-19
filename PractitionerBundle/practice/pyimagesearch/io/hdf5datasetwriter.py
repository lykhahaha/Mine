import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, output_path, dims, data_key='images', buf_size=1000):
        if os.path.exists(output_path):
            raise ValueError('The output_path is not existed', output_path)
        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset(data_key, dims, dtype='float')
        self.labels = self.db.create_dataset('labels', (dims[0], ), dtype='int')
        self.buf_size = buf_size
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    def add(self, rows, labels):
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)

        if len(self.buffer['labels']) >= self.buf_size:
            self.flush()

    def flush(self):
        idx_next = self.idx + len(self.buffer['labels'])
        self.data[self.idx:idx_next] = self.buffer['data']
        self.labels[self.idx:idx_next] = self.buffer['labels']

        self.idx = idx_next
        self.buffer = {'data': [], 'labels': []}

    def storeClassLabels(self, label_names):
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset('label_names', (len(label_names), ), dtype=dt)
        label_set[:] = label_names

    def close(self):
        if len(self.buffer['labels']) > 0:
            self.flush()

        self.db.close()