import torch
from torch import nn
import math


class GELU(nn.Module):
    """
    Approximation of GELU from p2 of the paper: https://arxiv.org/pdf/1606.08415.pdf.
    """

    def forward(self, X):
        return (
            0.5
            * X
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (X + 0.044715 * torch.pow(X, 3)))
            )
        )
