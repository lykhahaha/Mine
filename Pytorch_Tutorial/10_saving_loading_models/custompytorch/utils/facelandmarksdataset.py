from torch.utils.data import Dataset
import pandas as pd
from os import path
from skimage import io

class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_csv)
    
    def __getitem__(self, idx):
        image_path = path.sep.join([self.root_dir, self.landmarks_csv.iloc[idx, 0]])
        image = io.imread(image_path)
        landmarks = self.landmarks_csv.iloc[idx, 1:].values
        landmarks = landmarks.astype('float').reshape((-1, 2))
        sample = {
            'image': image,
            'landmarks': landmarks
        }

        if self.transform:
            sample = self.transform(sample)

        return sample