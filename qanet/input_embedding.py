import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from qanet.highway import Highway

class WordEmbedding(nn.Module):
    def __init__(self, word_embeddings):
        super(WordEmbedding, self).__init__()

        self.word_embedding = nn.Embedding(num_embeddings=word_embeddings.shape[0],
                                           embedding_dim=word_embeddings.shape[1])

        self.word_embedding.weight = nn.Parameter(torch.from_numpy(word_embeddings).float())
        self.word_embedding.weight.requires_grad = False

    def forward(self, input_context, input_question):
        context_word_emb = self.word_embedding(input_context)
        context_word_emb = F.dropout(context_word_emb, p=0.1, training=self.training)

        question_word_emb = self.word_embedding(input_question)
        question_word_emb = F.dropout(question_word_emb, p=0.1, training=self.training)

        return context_word_emb, question_word_emb


class CharacterEmbedding(nn.Module):
    def __init__(self, char_embeddings, embedding_dim=32, n_filters=200, kernel_size=5, padding=2):
        super(CharacterEmbedding, self).__init__()

        self.num_embeddings = len(char_embeddings)
        self.embedding_dim = embedding_dim
        self.kernel_size = (1, kernel_size)
        self.padding = (0, padding)

        self.char_embedding = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=embedding_dim)
        self.char_embedding.weight = nn.Parameter(torch.from_numpy(char_embeddings).float())

        self.char_conv = nn.Conv2d(in_channels=embedding_dim,
                                   out_channels=n_filters,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding)

    def forward(self, x):
        batch_size = x.shape[0]
        word_length = x.shape[-1]

        x = x.view(batch_size, -1)
        x = self.char_embedding(x)
        x = x.view(batch_size, -1, word_length, self.embedding_dim)

        # embedding dim of characters is number of channels of conv layer
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.char_conv(x))
        x = x.permute(0, 2, 3, 1)

        # max pooling over word length to have final tensor
        x, _ = torch.max(x, dim=2)

        x = F.dropout(x, p=0.05, training=self.training)

        return x


class InputEmbedding(nn.Module):
    def __init__(self, word_embeddings, char_embeddings, word_embed_dim=300,
                 char_embed_dim=32, char_embed_n_filters=200,
                 char_embed_kernel_size=7, char_embed_pad=3, highway_n_layers=2, hidden_size=128):

        super(InputEmbedding, self).__init__()

        self.word_embedding = WordEmbedding(word_embeddings)
        self.character_embedding = CharacterEmbedding(char_embeddings,
                                                     embedding_dim=char_embed_dim,
                                                     n_filters=char_embed_n_filters,
                                                     kernel_size=char_embed_kernel_size,
                                                     padding=char_embed_pad)


        # ========================== cove start ==============================
        state_dict = torch.load('data/MT-LSTM.pt')

        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True, batch_first=True)

        state_dict1 = dict([(name, param.data) if isinstance(param, Parameter) else (name, param)
                         for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([(name.replace('1', '0'), param.data) if isinstance(param, Parameter) else (name.replace('1', '0'),param) for name, param in state_dict.items() if '1' in name])
        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)
        for p in self.parameters(): p.requires_grad = False

        # ========================== cove end ================================

        self.projection = nn.Conv1d(word_embed_dim + char_embed_n_filters + 600, hidden_size, 1)

        self.highway = Highway(input_size=hidden_size, n_layers=highway_n_layers)

    def forward(self, context_word, context_char, question_word, question_char):
        
        context_word, question_word = self.word_embedding(context_word, question_word)
        context_char = self.character_embedding(context_char)
        question_char = self.character_embedding(question_char)

        context = torch.cat((context_word, context_char), dim=-1)
        question = torch.cat((question_word, question_char), dim=-1)
        
        context_low_cove, _ = self.rnn1(context_word)
        context_high_cove, _ = self.rnn2(context_low_cove)
        question_low_cove, _ = self.rnn1(question_word)
        question_high_cove, _ = self.rnn2(question_low_cove)
        
        context = self.projection(torch.cat([context, context_high_cove], dim=-1).permute(0, 2, 1))
        question = self.projection(torch.cat([question, question_high_cove], dim=-1).permute(0, 2, 1))

        context = self.highway(context)
        question = self.highway(question)

        return context, question
