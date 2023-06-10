import torch

from dldemos.Transformer.data_load import (idx_to_sentence, load_cn_vocab,
                                           load_en_vocab, maxlen)
from dldemos.Transformer.model import Transformer

# Config
batch_size = 1
lr = 0.0001
d_model = 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.2
n_epochs = 60

PAD_ID = 0


def main():
    device = 'cuda'
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    model = Transformer(len(en2idx), len(cn2idx), 0, d_model, d_ff, n_layers,
                        heads, dropout_rate, maxlen)
    model.to(device)
    model.eval()

    model_path = 'dldemos/Transformer/model.pth'
    model.load_state_dict(torch.load(model_path))

    my_input = ['we', 'should', 'protect', 'environment']
    x_batch = torch.LongTensor([[en2idx[x] for x in my_input]]).to(device)

    cn_sentence = idx_to_sentence(x_batch[0], idx2en, True)
    print(cn_sentence)

    y_input = torch.ones(batch_size, maxlen,
                         dtype=torch.long).to(device) * PAD_ID
    y_input[0] = en2idx['<S>']
    # y_input = y_batch
    with torch.no_grad():
        for i in range(1, y_input.shape[1]):
            y_hat = model(x_batch, y_input)
            for j in range(batch_size):
                y_input[j, i] = torch.argmax(y_hat[j, i - 1])
    output_sentence = idx_to_sentence(y_input[0], idx2cn, True)
    print(output_sentence)


if __name__ == '__main__':
    main()
