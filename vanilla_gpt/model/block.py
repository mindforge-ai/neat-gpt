from torch import nn


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
