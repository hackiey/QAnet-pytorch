
import torch.nn as nn
from qanet.encoder_block import EncoderBlock

class ModelEncoder(nn.Module):
    def __init__(self, n_blocks=7, n_conv=2, kernel_size=7, padding=3,
                 hidden_size=128, conv_type='depthwise_separable', n_heads=8, context_length=400):
        
        super(ModelEncoder, self).__init__()

        self.n_conv = n_conv
        self.n_blocks = n_blocks
        self.total_layers = (n_conv + 2) * n_blocks

        self.stacked_encoderBlocks = nn.ModuleList([EncoderBlock(n_conv=n_conv,
                                                                kernel_size=kernel_size,
                                                                padding=padding,
                                                                n_filters=hidden_size,
                                                                conv_type=conv_type,
                                                                n_heads=n_heads) for i in range(n_blocks)])

    def forward(self, x, mask):
        
        for i in range(self.n_blocks):
            x = self.stacked_encoderBlocks[i](x, mask, i*(self.n_conv+2)+1, self.total_layers)
        M0 = x

        for i in range(self.n_blocks):
            x = self.stacked_encoderBlocks[i](x, mask, i*(self.n_conv+2)+1, self.total_layers)
        M1 = x

        for i in range(self.n_blocks):
            x = self.stacked_encoderBlocks[i](x, mask, i*(self.n_conv+2)+1, self.total_layers)

        M2 = x

        return M0, M1, M2
