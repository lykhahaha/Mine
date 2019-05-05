import torch
from torch import nn, optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from custompytorch.utils.helpers import train_model, visualize_model

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

# to train a model to classify ants and bees. We have about 120 training images each for ants and bees. There are 75 validation images for each class
# wget https://download.pytorch.org/tutorial/hymenoptera_data.zip

# Data augmentation and normalization for training, just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}
IMAGE_PATHS_ROOT = os.path.sep.join(['..', 'datasets', 'hymenoptera_data'])
image_paths = {dtype: datasets.ImageFolder(os.path.sep.join([IMAGE_PATHS_ROOT, dtype]), transform=data_transforms[dtype]) for dtype in data_transforms.keys()}
dataloaders = {dtype: torch.utils.data.DataLoader(image_paths[dtype], batch_size=64, shuffle=True if dtype=='train' else False, num_workers=4) for dtype in image_paths.keys()}
dataset_sizes = {dtype: len(image_paths[dtype]) for dtype in image_paths.keys()}
class_names = image_paths['train'].classes

# Finetuning the model
model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.fc = nn.Linear(model_ft.fc.in_features, len(class_names))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
epochs = 25

# Apply Learning rate scheduler by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=epochs)

visualize_model(model_ft, dataloaders, class_names, file_names='finetuning_prediction_sample.jpg', num_images=6)