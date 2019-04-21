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

# define criterion and optimizer
optimizer = optim.SGD((w1, w2), lr=lr)

print('[INFO] Training on Pytorch tensors with Autograd')

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
# -------------------------------------------------------------------------------------
#-------------------------Pytorch: custom autograd------------------------------------------------------------
import torch
from torch import optim
import torch.nn.functional as F
from custompytorch.autograd import MyReLU

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

print('[INFO] Training on Pytorch tensors with custom autograd')

for e in range(epochs):
    # To apply our Function, we use Function.apply
    relu = MyReLU.apply
    
    optimizer.zero_grad()

    # Forward pass: compute predicted y using operations; we compute ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss. Now loss is Tensor of shape(1, ), loss.item() gets scalar value
    loss = (y_pred - y).pow(2).sum()
    if (e+1) % 10 == 0:
        print(f'[INFO] epoch {e+1}/{epochs}: Loss: {loss.item():.5f}')

    loss.backward()
    optimizer.step()
#--------------------------------------------------------------------------------------------------------------
#-------------------Compare between static graph of Tensorflow and dynamic graph of Pytorch--------------------
import tensorflow as tf
import numpy as np

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# create placeholders for input and target data, these will be filled with real data when executing graph
x, y = tf.placeholder(tf.float32, shape=(None, depth_in)), tf.placeholder(tf.float32, shape=(None, depth_out))

# Create Variables for weights and initialize them with random data. a TF Variable persists its value across executions of the graph
w1, w2 = tf.Variable(tf.random_normal((depth_in, hidden_dim))), tf.Variable(tf.random_normal((hidden_dim, depth_out)))

# define forward pass
hidden_layer = tf.matmul(x, w1)
hidden_layer_relu = tf.maximum(hidden_layer, tf.zeros(1))
y_pred = tf.matmul(hidden_layer_relu, w2)

# Compute loss using operations on tf tensors
loss = tf.reduce_sum((y_pred - y) ** 2.0)

# Compute gradient of loss wrt w1, w2
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

# Update weights using gradient descent. To actually update weights,we need to evaluate new_w1, new_w2 when executing the graph
# Note that in TF, the act of updating value of weights is part of computational graph; in Pytorch this happens outside the computational graph
lr = 1e-6
new_w1 = w1.assign(w1 - lr * grad_w1)
new_w2 = w2.assign(w2 - lr * grad_w2)

# Now we built our computationally graph, so we enter TF session to actually execute graph
with tf.Session() as sess:
    # Rub the graph once to initialize variables w1 and w2
    sess.run(tf.global_variables_initializer())

    # Create numpy arrays holding actual data for the inputs x and targets y
    x_value, y_value = np.random.randn(batch_size, depth_in), np.random.randn(batch_size, depth_out)

    for e in range(epochs):
        # Execute graph many times. Each time it executes, we want to bind x_value to x and y_value to y, specified with feed_dict argument.
        # Each time we execute graph, we want to compute values for loss=, the values of the Tensors are numpy arrays
        loss_value, _, _ = sess.run([loss, new_w1, new_w2], feed_dict={x: x_value, y: y_value})

        if (e+1) % 10 == 0:
            print(f'[INFO] epoch {e+1}/{epochs}: Loss: {loss_value:.5f}')
#-----------------------------------------------------------------------------------------------------------------
#----------------------------------------------Pytorch: nn------------------------------------------------------
import torch
from torch import nn

device = torch.device('cpu') # torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# Create random in and output
x, y = torch.randn(batch_size, depth_in, device=device, dtype=torch.float), torch.randn(batch_size, depth_out, device=device, dtype=torch.float)

# Use nn package to define our model as a sequence of layers
# nn.Sequentail() is a Module which contains other Modules, and applies them in sequence to produce output
# Linear Module holds internal Tensors for its weight and bias
model = nn.Sequential(
    nn.Linear(depth_in, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, depth_out)
)

# nn package also contains definitions of popular loss functions; in this case we will use Mean Square Error (MSE) as loss
criterion = nn.MSELoss(reduction='sum')

lr = 1e-4
epochs = 500

print('[INFO] Training on Pytorch tensors with nn Module')

for e in range(epochs):
    # In forward, predict y is computed by passing x to model. Module objects override the __call__ operator so you can call them like functions
    y_pred = model(x)

    # Compute nad print loss
    loss = criterion(y_pred, y)
    if (e+1) % 10 == 0:
        print(f'[INFO] epoch {e+1}/{epochs}: Loss: {loss.item():.5f}')
    
    # Xero out the gradients before running backward pass
    model.zero_grad()

    # Backward: compute gradient of the loss wrt all learnable parameters of the model
    # Internally, the parameters of each Module are stored in Tensors with requires_grad=True, so this call will compute gradients for all learnable parameters in model
    loss.backward()

    # Update weights using gradient descent. Each parameters is a Tensor, so we can access its gradient like we did before
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad
#---------------------------------------------------------------------------------------------------------------
# ------------------------------------------------Pytorch: optim------------------------------------------------
import torch
from torch import optim
from torch import nn

device = torch.device('cpu') # torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# Create random in and output
x, y = torch.randn(batch_size, depth_in, device=device, dtype=torch.float), torch.randn(batch_size, depth_out, device=device, dtype=torch.float)

# Use nn package to define our model as a sequence of layers
model = nn.Sequential(
    nn.Linear(depth_in, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, depth_out)
)

# nn package also contains definitions of popular loss functions; in this case we will use Mean Square Error (MSE) as loss
criterion = nn.MSELoss(reduction='sum')

# Use optim package to define an Optimizer that will update the weights of model for us.
lr = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 500

print('[INFO] Training on Pytorch tensors with nn and optim Modules')

for e in range(epochs):
    # Forward pass: compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if (e+1) % 10 == 0:
        print(f'[INFO] epoch {e+1}/{epochs}: Loss: {loss.item():.5f}')
    
    # Before backward, use the optimizer object to zero out all of the gradients for the variables
    # This is because by default, gradients are accumulated in buffers(i.e, not overwritten) whenever .backward() is called
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss wrt model parameters
    loss.backward()

    # Calling step function on an Optimizer makes an update to its parameters
    optimizer.step()
# --------------------------------------------------------------------------------------------
#------------------------------Pytorch: Custom nn Module-------------------------------------------------------
import torch
from torch import nn
from torch import optim
from custompytorch.nn import TwoLayerNet

device = torch.device('cpu') # torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# Create random in and output
x, y = torch.randn(batch_size, depth_in, device=device, dtype=torch.float), torch.randn(batch_size, depth_out, device=device, dtype=torch.float)

# Construct model by instantiating custom nn Module
model = TwoLayerNet(depth_in, hidden_dim, depth_out)

# Define loss and optimizer used
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=1e-4)
epochs = 500

for e in range(epochs):
    # Forward
    y_pred = model(x)

    # Compute loss
    loss = criterion(y_pred, y)
    if (e+1) % 10 == 0:
        print(f'[INFO] epoch {e+1}/{epochs}: Loss: {loss.item():.5f}')
    
    # Zero out gradients, perform a backward pass and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#--------------------------------------------------------------------------------------------------------------
#-----------------------------------------Pytorch: dynamic graphs and weight sharing---------------------------
import torch
from torch import nn
from torch import optim
from custompytorch.nn import DynamicNet

device = torch.device('cpu') # torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

batch_size, depth_in, hidden_dim, depth_out = 64, 1000, 100, 10

# Create random in and output
x, y = torch.randn(batch_size, depth_in, device=device, dtype=torch.float), torch.randn(batch_size, depth_out, device=device, dtype=torch.float)

# Construct model by instantiating custom nn Module
model = DynamicNet(depth_in, hidden_dim, depth_out)

# Define loss and optimizer used
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
epochs = 500

for e in range(epochs):
    # Forward
    y_pred = model(x)

    # Compute loss
    loss = criterion(y_pred, y)
    if (e+1) % 10 == 0:
        print(f'[INFO] epoch {e+1}/{epochs}: Loss: {loss.item():.5f}')
    
    # Zero out gradients, perform a backward pass and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()