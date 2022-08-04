from torch import nn
from .block import Block


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding()
        self.position_embedding = nn.Embedding()
        self.stack = nn.ModuleList([Block() for _ in range(config.depth)])
        self.final_layernorm = nn.LayerNorm()
