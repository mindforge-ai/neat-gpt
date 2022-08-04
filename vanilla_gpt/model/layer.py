from torch import nn
from .multi_head_attention import MultiHeadAttention
from .mlp import MLP


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(config["embedding_dim"])
        self.attention = MultiHeadAttention(config)
        self.layernorm2 = nn.LayerNorm(config["embedding_dim"])
        self.mlp = MLP(config)

    def forward(self, X):
        residual = X
        X = self.layernorm1(X)
        X = self.attention(X)
        X = residual + X

        residual = X
        X = self.layernorm2(X)
        X = self.mlp(X)
        X = residual + X
        return X
