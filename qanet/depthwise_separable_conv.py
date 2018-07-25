
import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self, n_filters=128, kernel_size=7, padding=3):
        super(DepthwiseSeparableConv1d, self).__init__()

        self.depthwise = nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=padding, groups=n_filters)
        self.separable = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.separable(x)

        return x