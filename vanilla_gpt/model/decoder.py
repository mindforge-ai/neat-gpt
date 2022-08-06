import torch
from torch import nn
from .layer import Layer


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_len = config["context_len"]
        self.token_embedding = nn.Embedding(
            config["vocab_len"], config["embedding_dim"]
        )
        self.position_embedding = nn.Embedding(
            config["context_len"], config["embedding_dim"]
        )
        self.stack = nn.ModuleList([Layer(config) for _ in range(config["num_layers"])])

    def forward(self, X):
        embedded_tokens = self.token_embedding(X)

        position_indices = torch.arange(0, self.context_len).unsqueeze(0)
        embedded_positions = self.position_embedding(position_indices)

        hidden_states = embedded_tokens + embedded_positions

        for i, layer in enumerate(self.stack):
            hidden_states = layer(hidden_states)

        return hidden_states
