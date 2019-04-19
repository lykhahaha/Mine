#-------------------------------------Warm up: numpy-----------------------------------------------------
import numpy as np

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# Create random in and output
x, y = np.random.randn(batch_size, depth_in), np.random.randn(batch_size, depth_out)

# Create random weights
w1, w2 = np.random.randn(depth_in, hidden_dim), np.random.randn(hidden_dim, depth_out)

lr = 1e-6
epochs = 500

print('[INFO] Training on Numpy')

for e in range(epochs):
    # Forward prop
    hidden_layer = x.dot(w1)
    hidden_layer_relu = np.maximum(hidden_layer, 0)
    y_pred = hidden_layer_relu.dot(w2)

    # Compute loss
    loss = np.square(y_pred - y).sum()
    if (e+1) % 10 == 0:
        print(f'[INFO] epoch {e+1}/{epochs}: Loss: {loss:.5f}')

    # Backprop
    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = hidden_layer.T.dot(grad_y_pred)
    grad_hidden_layer_relu = grad_y_pred.dot(w2.T)
    grad_hidden_layer = grad_hidden_layer_relu * np.where(hidden_layer_relu==0, 0, 1)
    grad_w1 = x.T.dot(grad_hidden_layer)

    # update weights
    w2 -= lr * grad_w2
    w1 -= lr * grad_w1

# -------------------------------------------------------------------------------------------------------

# ---------------------------------Pytorch: Tensors without Autograd-------------------------------------
import torch

device = torch.device('cpu') # torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# Create random in and output
x, y = torch.randn(batch_size, depth_in, device=device, dtype=torch.float), torch.randn(batch_size, depth_out, device=device, dtype=torch.float)

# Create random weights
w1, w2 = torch.randn(depth_in, hidden_dim, device=device, dtype=torch.float), torch.randn(hidden_dim, depth_out, device=device, dtype=torch.float)

lr = 1e-6
epochs = 500

print('[INFO] Training on Pytorch tensors without Autograd')

for e in range(epochs):
    # Forward prop
    hidden_layer = x.mm(w1)
    hidden_layer_relu = hidden_layer.clamp(min=0)
    y_pred = hidden_layer_relu.mm(w2)

    # Compute loss
    loss = (y_pred - y).pow(2).sum().item()
    if (e+1) % 10 == 0:
        print(f'[INFO] epoch {e+1}/{epochs}: Loss: {loss:.5f}')

    # Backprop
    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = hidden_layer.t().mm(grad_y_pred)
    grad_hidden_layer_relu = grad_y_pred.mm(w2.t())
    grad_hidden_layer = grad_hidden_layer_relu * torch.where(hidden_layer_relu==0, torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32))
    grad_w1 = x.t().mm(grad_hidden_layer)

    # update weights
    w2 -= lr * grad_w2
    w1 -= lr * grad_w1

# -------------------------------------------------------------------------------------
# ---------------------------------Pytorch: Tensors with Autograd-------------------------------------
import torch
from torch import optim

device = torch.device('cpu') # torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# Create random in and output
# Setting requires_grad=False indicates we do not need to compute gradients with respect to these Tensors during backward pass
x, y = torch.randn(batch_size, depth_in, device=device, dtype=torch.float), torch.randn(batch_size, depth_out, device=device, dtype=torch.float)

# Create random weights
# Setting requires_grad=True indicates we want to compute gradients with respect to these Tensors during backward pass
w1, w2 = torch.randn(depth_in, hidden_dim, device=device, dtype=torch.float, requires_grad=True), torch.randn(hidden_dim, depth_out, device=device, dtype=torch.float, requires_grad=True)

lr = 1e-6
epochs = 500

optimizer = optim.SGD((w1, w2), lr=lr)

for e in range(epochs):
    # Forward pass: compute predicted y using operations on Tensors, same operations we used to compute the forward pass using Tensors
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss. Now loss is Tensor of shape(1, ), loss.item() gets scalar value
    loss = (y_pred - y).pow(2).sum()
    if (e+1) % 10 == 0:
        print(f'[INFO] epoch {e+1}/{epochs}: Loss: {loss.item():.5f}')

    # Use autograd to compute backward pass. This call will compute gradient of loss wrt all Tensors with requires_grad=True
    # After this call w1.grad and w2.grad will be Tensors holding gradient of loss wrt w1 and w2 respectively
    loss.backward()

    # Manually update weights using gradient descent 
    # Wrap in torch.no_grad b/c weights has requires_grad=True, but we don't need to track this in autograd
    # Alternative way is to operate on weight.data and weight.grad.data. Recall that tensor.data gives a tensor that shares the storage with tensor, but doesn't track history
    # Can also use torch.optim.SGD()

    # optimizer.step()
    # optimizer.zero_grad()

    # OR

    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        # Manually zero gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()