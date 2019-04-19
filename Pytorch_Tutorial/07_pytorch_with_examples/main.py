#-------------------------------------Warm up: numpy-----------------------------------------------------
import numpy as np

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# Create random in and output
x, y = np.random.randn(batch_size, depth_in), np.random.randn(batch_size, depth_out)

# Create random weights
w1, w2 = np.random.randn(depth_in, hidden_dim), np.random.randn(hidden_dim, depth_out)

lr = 1e-6
epochs = 500

for e in range(epochs):
    # Forward prop
    hidden_layer = x.dot(w1)
    hidden_layer_relu = np.maximum(hidden_layer, 0)
    y_pred = hidden_layer_relu.dot(w2)

    # Compute loss
    loss = np.square(y_pred - y).sum()
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

# ---------------------------------Pytorch: Tensors------------------------------------------------------
import torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# Create random in and output
x, y = torch.randn(batch_size, depth_in, device=device, dtype=torch.float), torch.randn(batch_size, depth_out, device=device, dtype=torch.float)

# Create random weights
w1, w2 = torch.randn(depth_in, hidden_dim, device=device, dtype=torch.float), torch.randn(hidden_dim, depth_out, device=device, dtype=torch.float)