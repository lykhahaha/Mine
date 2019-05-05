import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import optim, nn
from custompytorch.nn import Mnist
from custompytorch.utils import helpers

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

data, targets = load_digits().data, load_digits().target
train_x, val_x, train_y, val_y = train_test_split(data, targets, test_size=0.15, random_state=42)

train_x, val_x = map(lambda x: torch.tensor(x, dtype=torch.float32), (train_x, val_x))
train_y, val_y = map(lambda x: torch.tensor(x, dtype=torch.int64), (train_y, val_y))
trainset, valset = TensorDataset(train_x, train_y), TensorDataset(val_x, val_y)
trainloader, valloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2), DataLoader(valset, batch_size=64, num_workers=2)
dataloaders = {'train': trainloader, 'val': valloader}
dataset_sizes = {'train': len(train_x), 'val': len(val_x)}

model = Mnist()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
epochs = 25

model = helpers.train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=epochs)
