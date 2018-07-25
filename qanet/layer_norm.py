import torch
import torch.nn as nn

class LayerNorm1d(nn.Module):

    def __init__(self, n_features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_features))
        self.beta = nn.Parameter(torch.zeros(n_features))
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 1)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.gamma * (x - mean) / (std + self.eps) + self.beta
        return x.permute(0, 2, 1)