# import ipdb
import os
import pickle
import json
import math
import torch
import torch.nn.functional as F
import pickle
import torcheras
import argparse
import numpy as np

# from torchsummary import summary
from collections import Counter
from torch.utils.data import DataLoader

from qanet.qanet import QANet
from dataset import QANetDataset
from constants import device
from utils import convert_tokens, evaluate

parser = argparse.ArgumentParser(description='save description')
parser.add_argument('description', default='')

criterion = torch.nn.CrossEntropyLoss()

def loss_function(y_pred, y_true):
    span = y_true[0]

    loss = criterion(y_pred[0], span[:, 0])
    loss += criterion(y_pred[1], span[:, 1])
    return loss

def count_parameters(model):
    parameters = [p for p in model.parameters() if p.requires_grad]
    counts = [p.numel() for p in parameters]

    for p, c in zip(parameters, counts):
        print(p.shape, c)

    return sum(counts)

def train(params, description):
    train_dataset = QANetDataset('data', 'train')
    dev_dataset = QANetDataset('data', 'dev')

    train_eval = pickle.load(open('data/train_eval.pkl', 'rb'))
    dev_eval = pickle.load(open('data/dev_eval.pkl', 'rb'))

    def evaluate_em(y_true, y_pred):
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

        if y_pred[0].requires_grad:
            answer_dict, _ = convert_tokens(
                train_eval, qa_id.tolist(), p1, p2)
            metrics = evaluate(train_eval, answer_dict)
        else:
            answer_dict, _ = convert_tokens(
                dev_eval, qa_id.tolist(), p1, p2)
            metrics = evaluate(dev_eval, answer_dict)

        return torch.Tensor([metrics['exact_match']])

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=params['batch_size'], shuffle=True)

    word_emb_mat = np.array(pickle.load(open(os.path.join(params['target_dir'], 'word_emb_mat.pkl'), 'rb')),
                            dtype=np.float32)
    char_emb_mat = np.array(pickle.load(open(os.path.join(params['target_dir'], 'char_emb_mat.pkl'), 'rb')),
                            dtype=np.float32)

    qanet = QANet(params, word_emb_mat, char_emb_mat).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, qanet.parameters()),
                                 lr=params['learning_rate'], betas=(params['beta1'], params['beta2']),
                                 weight_decay=params['weight_decay'])
    crit = 1 / math.log(1000)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda ee: crit * math.log(ee + 1) if (
                                                                                                  ee + 1) <= 1000 else 1)

    qanet = torcheras.Model(qanet, 'log/qanet')

    print(description)
    qanet.set_description(description)

    custom_objects = {'em': evaluate_em}
    qanet.compile(loss_function, scheduler, metrics=['em'], device=device, custom_objects=custom_objects)
    qanet.fit(train_loader, dev_loader, ema_decay=0.9999, grad_clip=5)


if __name__ == '__main__':
    args = parser.parse_args()

    params = json.load(open('params.json', 'r'))
    train(params, args.description)
