import torch
import torch.nn as nn

from dldemos.Transformer.preprocess_data import get_dataloader, load_vocab, load_sentences
from dldemos.Transformer.dataset import tensor_to_sentence
from dldemos.Transformer.model import Transformer


def main():
    en_vocab, zh_vocab = load_vocab()

    en_train, zh_train, en_valid, zh_valid = load_sentences()
    dataloader_train = get_dataloader(en_train, zh_train)
    dataloader_valid = get_dataloader(en_valid, zh_valid)

    device = 'cuda:0'

    # model config
    lr = 0.0001
    d_model = 256
    d_ff = 1024
    n_layers = 6
    heads = 8
    model = Transformer(len(en_vocab), len(zh_vocab), d_model, d_ff, n_layers,
                        heads)

    optimizer = torch.optim.Adam(model.parameters(), lr)
    citerion = nn.CrossEntropyLoss()
    for i in range(10):
        loss_sum = 0
        dataset_len = len(dataloader_train.dataset)

        for x, y in dataloader_train:
            x, y = x.to(device), y.to(device)
            seq_len = x
            hat_y = model(y)
            n, Tx, _ = hat_y.shape
            hat_y = torch.reshape(hat_y, (n * Tx, -1))
            y = torch.reshape(y, (n * Tx, -1))
            label_y = torch.argmax(y, 1)
            loss = citerion(hat_y, label_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

    torch.save(model.state_dict(), 'dldemos/BasicRNN/rnn1.pth')


if __name__ == '__main__':
    main()
