import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dldemos.Transformer.dataset import (EOS_ID, PAD_ID, SOS_ID, create_vocab,
                                         read_file, sentence_to_tensor,
                                         tensor_to_sentence)


def load_vocab(filename='data/translation2019zh/vocab_30k_80k.npy'):
    vocab = np.load(filename, allow_pickle=True).item()
    en_vocab = vocab['en']
    zh_vocab = vocab['zh']
    return en_vocab, zh_vocab


def load_sentences(filename='data/translation2019zh/sentences.npy'):
    tensors = np.load(filename, allow_pickle=True).item()
    en_tensors_train = tensors['en_train']
    zh_tensors_train = tensors['zh_train']
    en_tensors_valid = tensors['en_valid']
    zh_tensors_valid = tensors['zh_valid']
    return (en_tensors_train, zh_tensors_train, en_tensors_valid,
            zh_tensors_valid)


class TranslationDataset(Dataset):

    def __init__(self, en_tensor: np.ndarray, zh_tensor: np.ndarray):
        super().__init__()
        assert len(en_tensor) == len(zh_tensor)
        self.length = len(en_tensor)
        self.en_tensor = en_tensor
        self.zh_tensor = zh_tensor

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = np.concatenate(([SOS_ID], self.en_tensor[index], [EOS_ID]))
        x = torch.from_numpy(x)
        y = np.concatenate(([SOS_ID], self.zh_tensor[index], [EOS_ID]))
        y = torch.from_numpy(y)
        return x, y


def get_dataloader(en_tensor: np.ndarray,
                   zh_tensor: np.ndarray,
                   batch_size=16,
                   dist_train=False):

    def collate_fn(batch):
        x, y = zip(*batch)
        x_pad = pad_sequence(x, batch_first=True, padding_value=PAD_ID)
        y_pad = pad_sequence(y, batch_first=True, padding_value=PAD_ID)

        return x_pad, y_pad

    dataset = TranslationDataset(en_tensor, zh_tensor)
    if dist_train:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                collate_fn=collate_fn)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
        return dataloader


def test1():
    # en_sens_train, zh_sens_train = read_file(
    #     'data/translation2019zh/translation2019zh_train.json')
    en_sens_valid, zh_sens_valid = read_file(
        'data/translation2019zh/translation2019zh_valid.json')
    en_vocab = create_vocab(en_sens_valid, 10000)
    zh_vocab = create_vocab(zh_sens_valid, 30000)

    en_tensors_valid = sentence_to_tensor(en_sens_valid, en_vocab)
    zh_tensors_valid = sentence_to_tensor(zh_sens_valid, zh_vocab)
    print(tensor_to_sentence(en_tensors_valid[1], en_vocab, True))
    print(tensor_to_sentence(zh_tensors_valid[1], zh_vocab))
    ds = TranslationDataset(en_tensors_valid, zh_tensors_valid)
    print(tensor_to_sentence(ds[1][0], en_vocab, True))
    print(tensor_to_sentence(ds[1][1], zh_vocab))
    dl = get_dataloader(en_tensors_valid, zh_tensors_valid)
    e, z = next(iter(dl))
    print(tensor_to_sentence(e[0], en_vocab, True))
    print(tensor_to_sentence(z[0], zh_vocab))


def test2():
    en_vocab, zh_vocab = load_vocab()

    en_train, zh_train, en_valid, zh_valid = load_sentences()
    dataloader_train = get_dataloader(en_train, zh_train)
    dataloader_valid = get_dataloader(en_valid, zh_valid)

    en_batch, zh_batch = next(iter(dataloader_train))
    print(tensor_to_sentence(en_batch[2], en_vocab, True))
    print(tensor_to_sentence(zh_batch[2], zh_vocab, False))

    en_batch, zh_batch = next(iter(dataloader_valid))
    print(tensor_to_sentence(en_batch[2], en_vocab, True))
    print(tensor_to_sentence(zh_batch[2], zh_vocab, False))


def main():

    en_sens_train, zh_sens_train = read_file(
        'data/translation2019zh/translation2019zh_train.json')
    en_sens_valid, zh_sens_valid = read_file(
        'data/translation2019zh/translation2019zh_valid.json')
    en_vocab = create_vocab(en_sens_train, 30000)
    zh_vocab = create_vocab(zh_sens_train, 80000)
    vocab = {'en': en_vocab, 'zh': zh_vocab}
    np.save('data/translation2019zh/vocab_30k_80k.npy', vocab)

    en_tensors_train = sentence_to_tensor(en_sens_train, en_vocab)
    zh_tensors_train = sentence_to_tensor(zh_sens_train, zh_vocab)
    en_tensors_valid = sentence_to_tensor(en_sens_valid, en_vocab)
    zh_tensors_valid = sentence_to_tensor(zh_sens_valid, zh_vocab)
    tensors = {
        'en_train': en_tensors_train,
        'zh_train': zh_tensors_train,
        'en_valid': en_tensors_valid,
        'zh_valid': zh_tensors_valid
    }
    np.save('data/translation2019zh/sentences.npy', tensors)


if __name__ == '__main__':
    # test1()
    # test2()
    main()
