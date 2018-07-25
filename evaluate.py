import os
import json
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torcheras

from dataset import QANetDataset
from constants import device

from qanet.qanet import QANet
from utils import convert_tokens, evaluate

def variable_data(sample_batched, device):
    x = sample_batched[0]
    y = sample_batched[1]
    if type(x) is list or type(x) is tuple:
        for i, _x in enumerate(x):
            x[i] = x[i].to(device)
    else:
        x = x.to(device)
    if type(y) is list or type(y) is tuple:
        for i, _y in enumerate(y):
            y[i] = y[i].to(device)
    else:
        y = y.to(device)

    return x, y

def evaluate_scores(y_true, y_pred, test_eval):
    qa_id = y_true[1]
    c_mask, q_mask = y_pred[2:]

    y_p1 = F.softmax(y_pred[0], dim=-1)
    y_p2 = F.softmax(y_pred[1], dim=-1)

    p1 = []
    p2 = []
    p_matrix = torch.bmm(y_p1.unsqueeze(2), y_p2.unsqueeze(1))
    for i in range(p_matrix.shape[0]):
        p = torch.triu(p_matrix[i])
        indexes = torch.argmax(p).item()
        p1.append(indexes // p.shape[0])
        p2.append(indexes % p.shape[0])

    answer_dict, _ = convert_tokens(
        test_eval, qa_id.tolist(), p1, p2)
    metrics = evaluate(test_eval, answer_dict)

    return metrics


def evaluate_model(params, dtype='test', model_folder='', model_epoch=''):
    test_dataset = QANetDataset('data', dtype)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    test_eval = pickle.load(open('data/' + dtype + '_eval.pkl', 'rb'))

    word_emb_mat = np.array(pickle.load(open(os.path.join(params['target_dir'], 'word_emb_mat.pkl'), 'rb')),
                            dtype=np.float32)
    char_emb_mat = np.array(pickle.load(open(os.path.join(params['target_dir'], 'char_emb_mat.pkl'), 'rb')),
                            dtype=np.float32)

    qanet = QANet(params, word_emb_mat, char_emb_mat).to(device)
    qanet = torcheras.Model(qanet, 'log/qanet')
    qanet.load_model(model_folder, epoch=model_epoch, ema=True)
    qanet = qanet.model
    qanet.eval()

    all_scores = {'em': 0, 'f1': 0}
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_dataloader):
            x, y_true = variable_data(sample_batched, device)
            y_pred = qanet(x)
            metrics = evaluate_scores(y_true, y_pred, test_eval)
            print(metrics)
            all_scores['em'] += metrics['exact_match']
            all_scores['f1'] += metrics['f1']

        print('em', all_scores['em'] / i_batch, 'f1', all_scores['f1'] / i_batch)

if __name__ == '__main__':
    params = json.load(open('params.json', 'r'))

    model_folder = '2018_7_24_13_45_8_514568'
    model_epoch = 25

    evaluate_model(params, dtype='test', model_folder=model_folder, model_epoch=model_epoch)
