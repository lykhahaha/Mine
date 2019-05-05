import torch
from torch import nn
from torch import optim
from custompytorch.nn import Cifar10Net

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

#-----------------------------------Define model, criterion, loss, optimizer---------------------------------------
# Initialize model
model = Cifar10Net()
model = model.to(device)

# Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 25

loss = criterion()
# model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(f'{param_tensor}\t{model.state_dict()[param_tensor].size()}')

# optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(f'{var_name}\t{optimizer.state_dict()[var_name]}')
#-------------------------------------------------------------------------------------------------------------------
#-------------------Saving and Loading Model for inference (save and load model.state_dict())-----------------------
torch.save(model.state_dict(), 'state_dict_cifar10.pth')

model = Cifar10Net()
model = model.to(device)
model.load_state_dict(torch.load('state_dict_cifar10.pth'))
model.eval()
#-------------------------------------------------------------------------------------------------------------------
#-----------------------------------------Saving and loading entire model-------------------------------------------
torch.save(model, 'cifar10.pth')
# Model must be defined somewhere
model = torch.load('cifar10.pth')
model = model.to(device)
model.eval()
#-------------------------------------------------------------------------------------------------------------------
#---------------------Saving and Loading a General Checkpoint for Inference/ Resuming training----------------------
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}, 'checkpoint_cifar10.tar')

model = Cifar10Net()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

checkpoint = torch.load('checkpoint_cifar10.tar')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epoch']
loss = checkpoint['loss']
#-------------------------------------------------------------------------------------------------------------------
#---------------------------Warmstarting Model Using Parameters from a Different Model------------------------------
