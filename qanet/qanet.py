import torch
import torch.nn as nn
import numpy as np

from qanet.input_embedding import InputEmbedding
from qanet.embedding_encoder import EmbeddingEncoder
from qanet.context_query_attention import ContextQueryAttention
from qanet.model_encoder import ModelEncoder
from qanet.output import Output

class QANet(nn.Module):
    ''' All-in-one wrapper for all modules '''

    def __init__(self, params, word_embeddings, char_embeddings):
        super(QANet, self).__init__()

        self.batch_size = params['batch_size']

        # Defining dimensions using data from the params.json file
        self.word_embed_dim = params['word_embed_dim']
        self.char_word_len = params["char_limit"]

        self.context_length = params['para_limit']
        self.question_length = params['ques_limit']

        self.char_embed_dim = params["char_dim"]
        self.char_embed_n_filters = params["char_embed_n_filters"]
        self.char_embed_kernel_size = params["char_embed_kernel_size"]
        self.char_embed_pad = params["char_embed_pad"]

        self.highway_n_layers = params["highway_n_layers"]

        self.hidden_size = params["hidden_size"]

        self.embed_encoder_resize_kernel_size = params["embed_encoder_resize_kernel_size"]
        self.embed_encoder_resize_pad = params["embed_encoder_resize_pad"]
        
        self.embed_encoder_n_blocks = params["embed_encoder_n_blocks"]
        self.embed_encoder_n_conv = params["embed_encoder_n_conv"]
        self.embed_encoder_kernel_size = params["embed_encoder_kernel_size"]
        self.embed_encoder_pad = params["embed_encoder_pad"]
        self.embed_encoder_conv_type = params["embed_encoder_conv_type"]
        self.embed_encoder_with_self_attn = params["embed_encoder_with_self_attn"]
        self.embed_encoder_n_heads = params["embed_encoder_n_heads"]

        self.model_encoder_n_blocks = params["model_encoder_n_blocks"]
        self.model_encoder_n_conv = params["model_encoder_n_conv"]
        self.model_encoder_kernel_size = params["model_encoder_kernel_size"]
        self.model_encoder_pad = params["model_encoder_pad"]
        self.model_encoder_conv_type = params["model_encoder_conv_type"]
        self.model_encoder_with_self_attn = params["model_encoder_with_self_attn"]
        self.model_encoder_n_heads = params["model_encoder_n_heads"]

        # Initializing model layers
        word_embeddings = np.array(word_embeddings)
        self.input_embedding = InputEmbedding(word_embeddings,
                                             char_embeddings,
                                             word_embed_dim=self.word_embed_dim,
                                             char_embed_dim=self.char_embed_dim,
                                             char_embed_n_filters=self.char_embed_n_filters,
                                             char_embed_kernel_size=self.char_embed_kernel_size,
                                             char_embed_pad=self.char_embed_pad,
                                             highway_n_layers=self.highway_n_layers,
                                             hidden_size=self.hidden_size)

        self.embedding_encoder = EmbeddingEncoder(resize_in=self.word_embed_dim + self.char_embed_n_filters,
                                                 hidden_size=self.hidden_size,
                                                 resize_kernel=self.embed_encoder_resize_kernel_size,
                                                 resize_pad=self.embed_encoder_resize_pad,
                                                 n_blocks=self.embed_encoder_n_blocks,
                                                 n_conv=self.embed_encoder_n_conv,
                                                 kernel_size=self.embed_encoder_kernel_size,
                                                 padding=self.embed_encoder_pad,
                                                 conv_type=self.embed_encoder_conv_type,
                                                 n_heads=self.embed_encoder_n_heads,
                                                 context_length=self.context_length,
                                                 question_length=self.question_length)

        self.context_query_attention = ContextQueryAttention(hidden_size=self.hidden_size)

        self.projection = nn.Conv1d(4 * self.hidden_size, self.hidden_size, kernel_size=1)

        self.model_encoder = ModelEncoder(n_blocks=self.model_encoder_n_blocks,
                                         n_conv=self.model_encoder_n_conv,
                                         kernel_size=self.model_encoder_kernel_size,
                                         padding=self.model_encoder_pad,
                                         hidden_size=self.hidden_size,
                                         conv_type=self.model_encoder_conv_type,
                                         n_heads=self.model_encoder_n_heads)
        self.output = Output(input_dim=self.hidden_size)

    def forward(self, x):
        context_word, context_char, question_word, question_char, c_mask, q_mask = x

        c_maxlen = int(c_mask.sum(1).max().item())
        q_maxlen = int(q_mask.sum(1).max().item())
        context_word = context_word[:, :c_maxlen]
        context_char = context_char[:, :c_maxlen, :]
        question_word = question_word[:, :q_maxlen]
        question_char = question_char[:, :q_maxlen, :]
        c_mask = c_mask[:, :c_maxlen]
        q_mask = q_mask[:, :q_maxlen]

        context_emb, question_emb = self.input_embedding(context_word, context_char, question_word, question_char)
        context_emb, question_emb = self.embedding_encoder(context_emb, question_emb, c_mask, q_mask)

        c2q_attn, q2c_attn = self.context_query_attention(context_emb, question_emb, c_mask, q_mask)
        mdl_emb = torch.cat((context_emb,
                  c2q_attn.permute(0, 2, 1),
                  context_emb*c2q_attn.permute(0, 2, 1),
                  context_emb*q2c_attn.permute(0, 2, 1)), 1)

        mdl_emb = self.projection(mdl_emb)

        M0, M1, M2 = self.model_encoder(mdl_emb, c_mask)

        p1, p2 = self.output(M0.permute(0,2,1), M1.permute(0,2,1), M2.permute(0,2,1))
    
        p1 = p1 - 1e30*(1-c_mask)
        p2 = p2 - 1e30*(1-c_mask)
    
        return p1, p2, c_mask, q_mask


if __name__ == '__main__':
    import json
    params = json.load(open('params.json', 'r'))
    qanet = QANet(params)

    print(dir(qanet))
