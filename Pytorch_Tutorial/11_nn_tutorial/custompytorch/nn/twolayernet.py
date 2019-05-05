import torch
from torch import nn

class TwoLayerNet(nn.Module):
    def __init__(self, depth_in, hidden_layer, depth_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as member variables.
        """
        super().__init__()
        self.linear1 = nn.Linear(depth_in, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, depth_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return a Tensor of output data.
        We can use Modules defined in the constructor as well as arbitrary operators on Tensors.
        """
        hidden_layer_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(hidden_layer_relu)
        return y_pred