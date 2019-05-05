from torch import nn

class Mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        return self.linear(x)