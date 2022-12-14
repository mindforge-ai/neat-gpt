from torch import nn
from .decoder import Decoder
import torch


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(config)
        self.clm_head = nn.Linear(
            config["embedding_dim"], config["vocab_len"], bias=False
        )
        self.clm_head.weight = (
            self.decoder.token_embedding.weight
        )  # weight tying, although I don't think this works fully.

    def forward(self, X):
        hidden_states = self.decoder(X)
        logits = self.clm_head(hidden_states)
        return logits
