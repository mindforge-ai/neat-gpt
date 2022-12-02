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


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = nn.Parameter(w)
            self.b = nn.Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x
