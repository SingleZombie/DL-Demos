import torch
import torch.nn as nn
import torch.distributed as dist
import time
import numpy as np

from dldemos.Transformer.preprocess_data import get_dataloader, load_vocab, load_sentences
from dldemos.Transformer.dataset import tensor_to_sentence
from dldemos.Transformer.model import Transformer

# Config
lr = 0.0001
d_model = 256
d_ff = 1024
n_layers = 6
heads = 8


def main():
    en_vocab, zh_vocab = load_vocab()

    en_train, zh_train, en_valid, zh_valid = load_sentences()
    dataloader_train = get_dataloader(en_train, zh_train)
    dataloader_valid = get_dataloader(en_valid, zh_valid)

    device = 'cuda:0'

    valid_period = 10

    model = Transformer(len(en_vocab), len(zh_vocab), d_model, d_ff, n_layers,
                        heads)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    citerion = nn.CrossEntropyLoss()
    tic = time.time()
    for epoch in range(3):
        loss_sum = 0
        dataset_len = len(dataloader_train.dataset)

        for x, y, x_mask, y_mask in dataloader_train:
            x, y, x_mask, y_mask = x.to(device), y.to(device), x_mask.to(
                device), y_mask.to(device)
            y_input = y[:, :-1]
            y_label = y[:, 1:]
            y_mask = y_mask[:, :-1]
            y_hat = model(x, y_input, x_mask, y_mask)
            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len, -1))
            y_label = torch.reshape(y_label, (n * seq_len, ))
            loss = citerion(y_hat, y_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss

        print(f'{tic // 60}:{ int(tic % 60)}')

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

        # if valid_period

    torch.save(model.state_dict(), 'dldemos/Transformer/model.pth')


if __name__ == '__main__':
    main()
