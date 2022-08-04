from torch import nn
from .utils import GELU


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inwards = nn.Linear(config["embedding_dim"], 4 * config["embedding_dim"])
        self.activation = GELU()
        self.outwards = nn.Linear(4 * config["embedding_dim"], config["embedding_dim"])

    def forward(self, X):
        X = self.inwards(X)
        X = self.activation(X)
        X = self.outwards(X)
        return X
