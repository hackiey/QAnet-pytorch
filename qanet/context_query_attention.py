import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextQueryAttention(nn.Module):

    def __init__(self, hidden_size=128):
        super(ContextQueryAttention, self).__init__()

        self.d = hidden_size

        self.W0 = nn.Linear(3 * self.d, 1)
        nn.init.xavier_normal_(self.W0.weight)

    def forward(self, C, Q, c_mask, q_mask):

        batch_size = C.shape[0]

        n = C.shape[2]
        m = Q.shape[2]

        q_mask.unsqueeze(-1)

        # Evaluate the Similarity matrix, S
        S = self.similarity(C.permute(0, 2, 1), Q.permute(0, 2, 1), n, m, batch_size)

        S_ = F.softmax(S - 1e30*(1-q_mask.unsqueeze(-1).permute(0, 2, 1).expand(batch_size, n, m)), dim=2)
        S__ = F.softmax(S - 1e30*(1-c_mask.unsqueeze(-1).expand(batch_size, n, m)), dim=1)

        A = torch.bmm(S_, Q.permute(0, 2, 1))
        #   AT = A.permute(0,2,1)
        B = torch.matmul(torch.bmm(S_, S__.permute(0, 2, 1)), C.permute(0, 2, 1))
        #   BT = B.permute(0,2,1)

        # following the paper, this layer should return the context2query attention
        # and the query2context attention
        return A, B

    def similarity(self, C, Q, n, m, batch_size):

        C = F.dropout(C, p=0.1, training=self.training)
        Q = F.dropout(Q, p=0.1, training=self.training)

        # Create QSim (#batch x n*m x d) where each of the m original rows are repeated n times
        Q_sim = self.repeat_rows_tensor(Q, n)
        # Create CSim (#batch x n*m x d) where C is reapted m times
        C_sim = C.repeat(1, m, 1)
        assert Q_sim.shape == C_sim.shape
        QC_sim = Q_sim * C_sim

        # The "learned" Similarity in 1 col, put back
        Sim_col = self.W0(torch.cat((Q_sim, C_sim, QC_sim), dim=2))
        # Put it back in right dim
        Sim = Sim_col.view(batch_size, m, n).permute(0, 2, 1)

        return Sim

    def repeat_rows_tensor(self, X, rep):
        (depth, _, col) = X.shape
        # Open dim after batch ("depth")
        X = torch.unsqueeze(X, 1)
        # Repeat the matrix in the dim opened ("depth")
        X = X.repeat(1, rep, 1, 1)
        # Permute depth and lines to get the repeat over lines
        X = X.permute(0, 2, 1, 3)
        # Return to input (#batch x #lines*#repeat x #cols)
        X = X.contiguous().view(depth, -1, col)

        return X

