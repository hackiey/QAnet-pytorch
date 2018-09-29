import torch
import torch.nn as nn
import torch.nn.functional as F

from qanet.position_encoding import PositionEncoding
from qanet.layer_norm import LayerNorm1d
from qanet.depthwise_separable_conv import DepthwiseSeparableConv1d
from qanet.self_attention import SelfAttention

class EncoderBlock(nn.Module):

    def __init__(self, n_conv, kernel_size=7, padding=3, n_filters=128, n_heads=8, conv_type='depthwise_separable'):
        super(EncoderBlock, self).__init__()

        self.n_conv = n_conv
        self.n_filters = n_filters

        self.position_encoding = PositionEncoding(n_filters=n_filters)

        # self.layer_norm = LayerNorm1d(n_features=n_filters)
        
        self.layer_norm = nn.ModuleList([LayerNorm1d(n_features=n_filters) for i in range(n_conv+2)])

        self.conv = nn.ModuleList([DepthwiseSeparableConv1d(n_filters,
                                                            kernel_size=kernel_size,
                                                            padding=padding) for i in range(n_conv)])
        self.self_attention = SelfAttention(n_heads, n_filters)

        self.fc = nn.Conv1d(n_filters, n_filters, kernel_size=1)

    def layer_dropout(self, inputs, residual, dropout):
        if self.training:
            if torch.rand(1) > dropout:
                outputs = F.dropout(inputs, p=0.1, training=self.training)
                return outputs + residual
            else:
                return residual
        else:
            return inputs + residual

    def forward(self, x, mask, start_index, total_layers):

        outputs = self.position_encoding(x)

        # convolutional layers
        for i in range(self.n_conv):
            residual = outputs
            outputs = self.layer_norm[i](outputs)

            if i % 2 == 0:
                outputs = F.dropout(outputs, p=0.1, training=self.training)
            outputs = F.relu(self.conv[i](outputs))

            # layer dropout
            outputs = self.layer_dropout(outputs, residual, (0.1 * start_index / total_layers))
            start_index += 1

        # self attention
        residual = outputs
        outputs = self.layer_norm[-2](outputs)

        outputs = F.dropout(outputs, p=0.1, training=self.training)
        outputs = outputs.permute(0, 2, 1)
        outputs = self.self_attention(outputs, mask)
        outputs = outputs.permute(0, 2, 1)

        outputs = self.layer_dropout(outputs, residual, 0.1 * start_index / total_layers)
        start_index += 1

        # fully connected layer
        residual = outputs
        outputs = self.layer_norm[-1](outputs)
        outputs = F.dropout(outputs, p=0.1, training=self.training)
        outputs = self.fc(outputs)
        outputs = self.layer_dropout(outputs, residual, 0.1 * start_index / total_layers)

        return outputs
