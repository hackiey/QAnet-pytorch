import os
import pickle
import numpy as np
from torch.utils.data import Dataset


class QANetDataset(Dataset):

    def __init__(self, data_dir, data_type):

        self.context_idxs = np.array(
            pickle.load(open(os.path.join(data_dir, data_type+'_context_idxs.pkl'), 'rb')), dtype=np.int64)
        self.context_char_idxs = np.array(
            pickle.load(open(os.path.join(data_dir, data_type+'_context_char_idxs.pkl'), 'rb')), dtype=np.int64)
        self.ques_idxs = np.array(
            pickle.load(open(os.path.join(data_dir, data_type+'_ques_idxs.pkl'), 'rb')), dtype=np.int64)
        self.ques_char_idxs = np.array(
            pickle.load(open(os.path.join(data_dir, data_type+'_ques_char_idxs.pkl'), 'rb')), dtype=np.int64)
        self.y = np.array(
            pickle.load(open(os.path.join(data_dir, data_type+'_y.pkl'), 'rb')), dtype=np.int64)
        self.ids = np.array(
            pickle.load(open(os.path.join(data_dir, data_type+'_ids.pkl'), 'rb')), dtype=np.int64)

    def __getitem__(self, index):
        context_idxs = self.context_idxs[index]
        ques_idxs = self.ques_idxs[index]

        c_mask = np.array(context_idxs > 0, dtype=np.float32)
        q_mask = np.array(ques_idxs > 0, dtype=np.float32)

        return (context_idxs, self.context_char_idxs[index],
               ques_idxs, self.ques_char_idxs[index], c_mask, q_mask), \
               (self.y[index], self.ids[index])

    def __len__(self):
        return len(self.context_idxs)