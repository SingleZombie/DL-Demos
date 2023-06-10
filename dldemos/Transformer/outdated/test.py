import torch
from dldemos.Transformer.outdated.preprocess_data import (get_dataloader,
                                                          load_vocab,
                                                          load_sentences,
                                                          SOS_ID, EOS_ID,
                                                          PAD_ID)
from dldemos.Transformer.dataset import tensor_to_sentence, MAX_SEQ_LEN
from dldemos.Transformer.model import Transformer

# Config
batch_size = 64
lr = 0.0001
d_model = 512
d_ff = 2048
n_layers = 6
heads = 8


def main():
    model_path = 'dldemos/Transformer/model_latest.pth'

    device = 'cuda'
    en_vocab, zh_vocab = load_vocab()

    en_train, zh_train, en_valid, zh_valid = load_sentences()
    dataloader_valid = get_dataloader(en_train, zh_train, 1)

    model = Transformer(len(en_vocab),
                        len(zh_vocab),
                        PAD_ID,
                        d_model,
                        d_ff,
                        n_layers,
                        heads,
                        max_seq_len=MAX_SEQ_LEN)
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    cnt = 0
    for x, y in dataloader_valid:
        x, y = x.to(device), y.to(device)
        x_mask = x == PAD_ID
        n = x.shape[0]
        sample = torch.ones(n, MAX_SEQ_LEN,
                            dtype=torch.long).to(device) * PAD_ID
        sample[:, 0] = SOS_ID
        print(tensor_to_sentence(x[0], en_vocab, True))
        print(tensor_to_sentence(y[0], zh_vocab))
        for i in range(50):
            sample_mask = sample == PAD_ID
            y_predict = model(x, sample, x_mask, sample_mask)
            y_predict = y_predict[:, i]
            prob_dist = torch.softmax(y_predict, 1)
            #new_word = torch.multinomial(prob_dist, 1)
            _, new_word = torch.max(prob_dist, 1)
            sample[:, i + 1] = new_word
            print(tensor_to_sentence(sample[0], zh_vocab))
        cnt += 1
        if cnt == 5:
            break

    print('Done.')


if __name__ == '__main__':
    main()
