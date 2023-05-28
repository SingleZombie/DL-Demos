import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import time
import numpy as np

from dldemos.Transformer.preprocess_data import (get_dataloader, load_vocab,
                                                 load_sentences, SOS_ID,
                                                 EOS_ID, PAD_ID)
from dldemos.Transformer.dataset import tensor_to_sentence, MAX_SEQ_LEN
from dldemos.Transformer.model import Transformer

# Config
batch_size = 64
lr = 0.0001
d_model = 256
d_ff = 1024
n_layers = 6
heads = 8


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def main():
    model_path = 'dldemos/Transformer/model_0.pth'

    device = 'cuda:0'
    en_vocab, zh_vocab = load_vocab()

    en_train, zh_train, en_valid, zh_valid = load_sentences()
    dataloader_valid = get_dataloader(en_valid, zh_valid)

    model = Transformer(len(en_vocab),
                        len(zh_vocab),
                        d_model,
                        d_ff,
                        n_layers,
                        heads,
                        max_seq_len=MAX_SEQ_LEN)
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    for x, y in dataloader_valid:
        x, y = x.to(device), y.to(device)
        x_mask = x == PAD_ID
        n = x.shape[0]
        sample = torch.ones(n, MAX_SEQ_LEN) * PAD_ID
        sample[:, 0] = SOS_ID
        for i in range(MAX_SEQ_LEN - 1):
            sample_mask = y == PAD_ID
            y_predict = model(x, sample, x_mask, sample_mask)
            y_predict = y_predict[:, i + 1]
            prob = torch.softmax(y_predict, 1)
            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len, -1))
            y_label = torch.reshape(y_label, (n * seq_len, ))
            loss = citerion(y_hat, y_label)

        # if valid_period

    print('Done.')


if __name__ == '__main__':
    main()
