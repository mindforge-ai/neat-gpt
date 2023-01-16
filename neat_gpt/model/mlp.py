from torch import nn
from .utils import GELU, Conv1D


class MLP(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # self.inwards = Conv1D(4 * embedding_dim, 1, embedding_dim)
        self.inwards = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.activation = GELU()
        # self.outwards = Conv1D(embedding_dim, 1, 4 * embedding_dim)
        self.outwards = nn.Linear(4 * embedding_dim, embedding_dim)

    def forward(self, X):
        X = self.inwards(X)
        X = self.activation(X)
        X = self.outwards(X)
        return X
