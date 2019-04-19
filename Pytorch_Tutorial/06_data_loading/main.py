import torch
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from pyimagesearch.utils import FaceLandmarksDataset

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='.')

class Resize(transforms.Resize):
    """Resize the face landmark input to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__(size)

    def __call__(self, face):
        image, landmarks = face['image'], face['landmarks']
        image_resized = super().__call__(Image.fromarray(image))
        image_resized = np.array(image_resized, dtype='uint8')
        
        landmarks = landmarks * [image_resized.shape[1]/image.shape[1], image_resized.shape[0]/image.shape[0]]
        return {'image': image_resized, 'landmarks': landmarks}

class RandomCrop():
    def __init__(self, size):
        assert isinstance(size, (int, tuple))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2
            self.size = size

    def __call__(self, face):
        image, landmarks = face['image'], face['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.size

        i, j = np.random.randint(0, h - new_h), np.random.randint(0, w - new_w)

        image = image[i:i + new_h, j:j + new_w]

        landmarks = landmarks - [j, i]

        return {'image': image, 'landmarks': landmarks}


class ToTensor:
    def __call__(self, face):
        image, landmarks = face['image'], face['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}

face_dataset = FaceLandmarksDataset(path.sep.join(['faces', 'face_landmarks.csv']), 'faces')

# plot 4 images without transform
fig = plt.figure()

for i, face in enumerate(face_dataset):
    print(i, face['image'].shape, face['landmarks'].shape)
    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title(f'Sample #{i}')
    ax.axis('off')
    show_landmarks(**face)

    if i == 3:
        plt.savefig('Data_loading_w_t_transform.png')
        plt.close(fig)
        break

# plot images with transform
transform = transforms.Compose([
    Resize(256),
    RandomCrop(224)
])

fig = plt.figure()
image_idx = np.random.randint(0, len(face_dataset))
face = face_dataset[image_idx]
for i, type_transform in enumerate([Resize(256), RandomCrop(224), transform]):
    face_transformed = type_transform(face)
    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    show_landmarks(**face_transformed)

plt.savefig('Data_loading_w_transform.png')
plt.close(fig)

transformed_face_dataset = FaceLandmarksDataset(csv_file=path.sep.join(['faces', 'face_landmarks.csv']),
                                    root_dir='faces',
                                    transform=transforms.Compose([
                                        Resize(256),
                                        RandomCrop(224),
                                        ToTensor()
                                    ]))

for i, transformed_face in enumerate(transformed_face_dataset):
    print(i, transformed_face['image'].size(), transformed_face['landmarks'].size())
    if i == 3:
        break

face_dataloader = torch.utils.data.DataLoader(transformed_face_dataset, batch_size=64, shuffle=True, num_workers=2)

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
age_dataset = torchvision.datasets.ImageFolder('age', transform=transform)
age_loader = torch.utils.data.DataLoader(age_dataset, batch_size=64,shuffle=True, num_workers=4) # labels is encoded 4, 5, 2, 0, 5, 0, 3