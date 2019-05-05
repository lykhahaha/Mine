import torch
from torch import optim, nn
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
from os import path
from custompytorch.utils import helpers

IMAGE_PATHS_ROOT = path.sep.join(['..', 'datasets', 'hymenoptera_data'])
MODEL_NAME = 'squeezenet'
NUM_CLASSES = 2
BATCH_SIZE = 64
INPUT_SIZE = 224
EPOCHS = 25

# Data augmentation and normalization for training, just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

print('[INFO] loading datasets...')
image_paths = {dtype: datasets.ImageFolder(path.sep.join([IMAGE_PATHS_ROOT, dtype]), transform=data_transforms[dtype]) for dtype in data_transforms.keys()}
dataloaders = {dtype: torch.utils.data.DataLoader(image_paths[dtype], batch_size=BATCH_SIZE, shuffle=True if dtype=='train' else False, num_workers=4) for dtype in data_transforms.keys()}
dataset_sizes = {dtype: len(image_paths[dtype]) for dtype in data_transforms.keys()}
class_names = image_paths['train'].classes

print('[INFO[ Finetuning the model...')
model, _ = helpers.initialize_model_finetuning(MODEL_NAME, NUM_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

model = helpers.train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=EPOCHS, is_inception=(MODEL_NAME=='inception'))