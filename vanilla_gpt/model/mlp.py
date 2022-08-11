from torch import nn
from .utils import GELU, Conv1D


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inwards = Conv1D(4 * config["embedding_dim"], 1, config["embedding_dim"])
        self.activation = GELU()
        self.outwards = Conv1D(config["embedding_dim"], 1, 4 * config["embedding_dim"])

    def forward(self, X):
        X = self.inwards(X)
        X = self.activation(X)
        X = self.outwards(X)
        return X
