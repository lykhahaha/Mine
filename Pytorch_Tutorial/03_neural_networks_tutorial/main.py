import torch
from torch import nn
from torch import optim

from pyimagesearch.nn.torchconv import LeNet

model = LeNet()
print(model)

params = list(model.parameters())
print(len(params))
print(params[0].size()) # conv1 weight
print(params[-1].size())

input_image = torch.randn(1, 1, 32, 32)
# input_image = torch.randn(1, 32, 32)
# input_image = input_image.unsqueeze(0) # 1, 1, 32, 32
prediction = model(input_image)
print(prediction)

model.zero_grad()
print(model.conv1.bias.grad)
# model.backward(torch.randn(1, 10)) # torch.randn(1, 10): initial gradient of last weight

target = torch.randn(1, 10)
criterion = nn.MSELoss()
loss = criterion(prediction, target)
loss.backward()
print(model.conv1.bias.grad)
# print(loss.grad_fn)
# print(loss.grad_fn.next_functions[0][0])
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])