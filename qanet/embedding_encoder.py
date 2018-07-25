
import torch.nn as nn
from qanet.encoder_block import EncoderBlock


class EmbeddingEncoder(nn.Module):
    def __init__(self, resize_in=500, hidden_size=128, resize_kernel=7, resize_pad=3,
                 n_blocks=1, n_conv=4, kernel_size=7, padding=3,
                 conv_type='depthwise_separable', n_heads=8, context_length=400, question_length=50):

        super(EmbeddingEncoder, self).__init__()

        self.n_conv = n_conv
        self.n_blocks = n_blocks
        self.total_layers = (n_conv+2)*n_blocks

        self.stacked_encoderBlocks = nn.ModuleList([EncoderBlock(n_conv=n_conv,
                                                                kernel_size=kernel_size,
                                                                padding=padding,
                                                                n_filters=hidden_size,
                                                                conv_type=conv_type,
                                                                n_heads=n_heads) for i in range(n_blocks)])

    def forward(self, context_emb, question_emb, c_mask, q_mask):
        for i in range(self.n_blocks):
            context_emb = self.stacked_encoderBlocks[i](context_emb, c_mask, i*(self.n_conv+2)+1, self.total_layers)
            question_emb = self.stacked_encoderBlocks[i](question_emb, q_mask, i*(self.n_conv+2)+1, self.total_layers)

        return context_emb, question_emb
