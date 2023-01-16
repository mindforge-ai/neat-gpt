from torch import nn
from .multi_head_attention import MultiHeadAttention
from .mlp import MLP


class Block(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_attention_heads,
        context_len,
        attention_dropout,
        outwards_dropout,
    ):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.attention = MultiHeadAttention(
            embedding_dim,
            num_attention_heads,
            context_len,
            attention_dropout,
            outwards_dropout,
        )
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim)

    def forward(self, X):
        residual = X
        X = self.attention(X)
        X = self.layernorm1(residual + X)

        residual = X
        X = self.mlp(X)
        X = self.layernorm2(residual + X)
        return X
