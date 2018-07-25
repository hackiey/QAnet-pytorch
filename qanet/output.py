import torch
import torch.nn as nn

class Output(nn.Module):
    def __init__(self, input_dim = 512):
        super(Output, self).__init__()

        self.d = input_dim

        self.W1 = nn.Linear(2*self.d, 1)
        self.W2 = nn.Linear(2*self.d, 1)

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)

    def forward(self, M0, M1, M2):

        p1 = self.W1(torch.cat((M0,M1), -1)).squeeze()
        p2 = self.W2(torch.cat((M0,M2), -1)).squeeze()
        return p1, p2
