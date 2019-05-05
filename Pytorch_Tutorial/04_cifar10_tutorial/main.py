import torch
import torchvision
from torchvision import transforms
from torch import optim
from torch import nn
from pyimagesearch.nn.torchconv  import Cifar10Net
from pyimagesearch.utils import AgeGenderHelper
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # images of range (0, 1)
])

# pip install tqdm==4.19.1
trainset = torchvision.datasets.CIFAR10(root='~/.torch/datasets', transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='~/.torch/datasets', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=2)

classes = np.array(['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Training on {device}...')

model = Cifar10Net()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

max_epoch = 5
for epoch in range(max_epoch):
    running_loss, running_val_loss = 0, 0
    labels_len, labels_val_len = 0, 0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        labels_len += labels.size(0)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item() * inputs.size(0)
            labels_val_len += labels.size(0)

    print(f'[INFO] Epoch {epoch+1}/{max_epoch}: Loss: {running_loss/labels_len:.5f}, Validation loss: {running_val_loss/labels_val_len:.5f}')

correct = 0
labels_len = 0
test_prediction, test_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        
        predicted = torch.argmax(outputs, 1)
        predicted = predicted.to('cpu')
        correct += (predicted == labels).sum().item()

        test_prediction.extend(predicted.tolist())
        test_labels.extend(labels.tolist())
        labels_len += labels.size(0)

print(f'[INFO]Accuracy of model over {labels_len} test images: {correct/labels_len:.5f}')

AgeGenderHelper.plot_confusion_matrix_from_transformed_data(classes[test_labels], classes[test_prediction], classes, 'cm_cifar10.jpg')