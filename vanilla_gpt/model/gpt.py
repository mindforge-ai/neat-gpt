from torch import nn
from .decoder import Decoder


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = Decoder(config)
        self.clm_head = nn.Linear(
            config["embedding_dim"], config["vocab_len"], bias=False
        )

    def forward(self, X):
        hidden_states = self.decoder(X)
        logits = self.clm_head(hidden_states)
        return logits
