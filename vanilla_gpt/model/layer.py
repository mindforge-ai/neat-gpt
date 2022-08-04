from torch import nn


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, X):
        return X
