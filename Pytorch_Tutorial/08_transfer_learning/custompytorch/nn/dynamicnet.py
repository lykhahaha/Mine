import torch
from torch import nn
import numpy as np

class DynamicNet(nn.Module):
    def __init__(self, depth_in, hidden_layer, depth_out):
        super().__init__()
        self.input_linear = nn.Linear(depth_in, hidden_layer)
        self.middle_linear = nn.Linear(hidden_layer, hidden_layer)
        self.output_linear = nn.Linear(hidden_layer, depth_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.
        """
        hidden_relu_layer = self.input_linear(x).clamp(min=0)
        for _ in range(np.random.randint(0, 4)):
            hidden_relu_layer = self.middle_linear(hidden_relu_layer).clamp(min=0)
        y_pred = self.output_linear(hidden_relu_layer)
        return y_pred