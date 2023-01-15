import torch
from torch import nn
from .block import Block


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_len = config["context_len"]
        self.token_embedding = nn.Embedding(
            config["vocab_len"], config["embedding_dim"]
        )
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        self.token_embedding_dropout = nn.Dropout(0.2)  # replace
        self.position_embedding = nn.Embedding(
            config["context_len"], config["embedding_dim"]
        )
        self.stack = nn.ModuleList([Block(config) for _ in range(config["num_layers"])])

    def forward(self, X):
        embedded_tokens = self.token_embedding(X)
        embedded_tokens = self.token_embedding_dropout(embedded_tokens)

        position_indices = torch.arange(0, self.context_len, device=X.device).unsqueeze(
            0
        )
        embedded_positions = self.position_embedding(position_indices)

        hidden_states = embedded_tokens + embedded_positions

        for i, layer in enumerate(self.stack):
            hidden_states = layer(hidden_states)

        return hidden_states
