import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from constants import device


class SelfAttention(nn.Module):

    def __init__(self, n_heads=8, n_filters=128):
        super(SelfAttention, self).__init__()

        self.n_filters = n_filters
        self.n_heads = n_heads

        self.key_dim = n_filters // n_heads
        self.value_dim = n_filters // n_heads

        self.fc_query = nn.ModuleList([nn.Linear(n_filters, self.key_dim) for i in range(n_heads)])
        self.fc_key = nn.ModuleList([nn.Linear(n_filters, self.key_dim) for i in range(n_heads)])
        self.fc_value = nn.ModuleList([nn.Linear(n_filters, self.value_dim) for i in range(n_heads)])
        self.fc_out = nn.Linear(n_heads * self.value_dim, n_filters)

    def forward(self, x, mask):
        batch_size = x.shape[0]
        l = x.shape[1]

        mask = mask.unsqueeze(-1).expand(x.shape[0], x.shape[1], x.shape[1]).permute(0,2,1)

        heads = torch.zeros(self.n_heads, batch_size, l, self.value_dim, device=device)

        for i in range(self.n_heads):
            Q = self.fc_query[i](x)
            K = self.fc_key[i](x)
            V = self.fc_value[i](x)

            # scaled dot-product attention
            tmp = torch.bmm(Q, K.permute(0,2,1))
            tmp = tmp / np.sqrt(self.key_dim)
            tmp = F.softmax(tmp - 1e30*(1-mask), dim=-1)

            tmp = F.dropout(tmp, p=0.1, training=self.training)

            heads[i] = torch.bmm(tmp, V)

        # concatenation is the same as reshaping our tensor
        x = heads.permute(1,2,0,3).contiguous().view(batch_size, l, -1)
        x = self.fc_out(x)

        return x


if __name__ == "__main__":
    batch_size = 8
    l = 60
    n_filters = 128

    mdl = SelfAttention()

    x = torch.ones(batch_size, l, n_filters)

    print(mdl(x))
