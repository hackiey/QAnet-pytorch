import torch
import numpy as np
import torch.nn as nn
from constants import device


class PositionEncoding(nn.Module):
    def __init__(self, n_filters=128, min_timescale=1.0, max_timescale=1.0e4):

        super(PositionEncoding, self).__init__()

        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.d = n_filters

        # we use the fact that cos(x) = sin(x + pi/2) to compute everything with one sin statement
        self.freqs = torch.Tensor(
            [max_timescale ** (-i / self.d) if i % 2 == 0 else max_timescale ** (-(i - 1) / self.d) for i in
             range(self.d)]).unsqueeze(1).to(device)
        self.phases = torch.Tensor([0 if i % 2 == 0 else np.pi / 2 for i in range(self.d)]).unsqueeze(1).to(device)

    def forward(self, x):

        # *************** speed up, static pos_enc ******************
        l = x.shape[-1]

        # computing signal
        pos = torch.arange(l, dtype=torch.float32).repeat(self.d, 1).to(device)
        tmp = pos * self.freqs + self.phases
        pos_enc = torch.sin(tmp)
        x = x + pos_enc

        return x


if __name__ == '__main__':
    mdl = PositionEncoding()

    batch_size = 8
    n_channels = 128
    n_items = 60

    input = torch.ones(batch_size, n_channels, n_items)

    out = mdl(input)
    print(out)
