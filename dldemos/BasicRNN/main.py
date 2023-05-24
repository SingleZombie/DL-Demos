from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from dldemos.BasicRNN.constant import EMBEDDING_LENGTH, LETTER_MAP
from dldemos.BasicRNN.models import RNN1, RNN2
from dldemos.BasicRNN.read_imdb import read_imdb_vocab, read_imdb_words


def words_to_label_array(words: Tuple[str, Sequence[str]], max_length):
    if isinstance(words, str):
        words = [words]
    words = [word + ' ' for word in words]
    batch = len(words)
    tensor = torch.zeros(batch, max_length, dtype=torch.long)
    for i in range(batch):
        for j, letter in enumerate(words[i]):
            tensor[i][j] = LETTER_MAP[letter]

    return tensor


def words_to_onehot(words: Tuple[str, Sequence[str]], max_length):
    if isinstance(words, str):
        words = [words]
    words = [word + ' ' for word in words]
    batch = len(words)
    tensor = torch.zeros(batch, max_length, EMBEDDING_LENGTH)
    for i in range(batch):
        word_length = len(words[i])
        for j in range(max_length):
            if j < word_length:
                tensor[i][j][LETTER_MAP[words[i][j]]] = 1
            else:
                tensor[i][j][0] = 1

    return tensor


def onehot_to_word(arr):
    len, emb_len = arr.shape
    out = []
    for i in range(len):
        for j in range(emb_len):
            if arr[i][j] == 1:
                out.append(j)
                break
    return out


class WordDataset(Dataset):

    def __init__(self, words, max_length, is_onehot=True):
        super().__init__()
        n_words = len(words)
        self.words = words
        self.n_words = n_words
        self.max_length = max_length
        self.is_onehot = is_onehot

    def __len__(self):
        return self.n_words

    def __getitem__(self, index):
        """return the (one-hot) encoding vector of a word."""
        word = self.words[index] + ' '
        word_length = len(word)
        if self.is_onehot:
            tensor = torch.zeros(self.max_length, EMBEDDING_LENGTH)
            for i in range(self.max_length):
                if i < word_length:
                    tensor[i][LETTER_MAP[word[i]]] = 1
                else:
                    tensor[i][0] = 1
        else:
            tensor = torch.zeros(self.max_length, dtype=torch.long)
            for i in range(word_length):
                tensor[i] = LETTER_MAP[word[i]]

        return tensor


def get_dataloader_and_max_length(limit_length=None,
                                  is_onehot=True,
                                  is_vocab=True):

    if is_vocab:
        words = read_imdb_vocab()
    else:
        words = read_imdb_words(n_files=200)

    max_length = 0
    for word in words:
        max_length = max(max_length, len(word))

    if limit_length is not None and max_length > limit_length:
        words = [w for w in words if len(w) <= limit_length]
        max_length = limit_length

    # for <EOS> (space)
    max_length += 1

    dataset = WordDataset(words, max_length, is_onehot)
    return DataLoader(dataset, batch_size=256), max_length


test_words = [
    'apple', 'appll', 'appla', 'apply', 'bear', 'beer', 'berr', 'beee', 'car',
    'cae', 'cat', 'cac', 'caq', 'query', 'queee', 'queue', 'queen', 'quest',
    'quess', 'quees'
]


def train_rnn1():
    device = 'cuda:0'
    dataloader, max_length = get_dataloader_and_max_length(19)

    model = RNN1().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):

        loss_sum = 0
        dataset_len = len(dataloader.dataset)

        for y in dataloader:
            y = y.to(device)
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
    return model


def train_rnn2():
    device = 'cuda:0'
    dataloader, max_length = get_dataloader_and_max_length(19, is_onehot=False)

    model = RNN2().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.CrossEntropyLoss()
    for epoch in range(5):

        loss_sum = 0
        dataset_len = len(dataloader.dataset)

        for y in dataloader:
            y = y.to(device)
            hat_y = model(y)
            n, Tx, _ = hat_y.shape
            hat_y = torch.reshape(hat_y, (n * Tx, -1))
            label_y = torch.reshape(y, (n * Tx, ))
            loss = citerion(hat_y, label_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

    torch.save(model.state_dict(), 'dldemos/BasicRNN/rnn2.pth')
    return model


def test_language_model(model, is_onehot=True, device='cuda:0'):
    _, max_length = get_dataloader_and_max_length(19)
    if is_onehot:
        test_word = words_to_onehot(test_words, max_length)
    else:
        test_word = words_to_label_array(test_words, max_length)
    test_word = test_word.to(device)
    probs = model.language_model(test_word)
    for word, prob in zip(test_words, probs):
        print(f'{word}: {prob}')


def sample(model):
    words = []
    for _ in range(20):
        word = model.sample_word()
        words.append(word)
    print(*words)


def rnn1():
    # rnn1 = train_rnn1()

    # Or load the models
    state_dict = torch.load('dldemos/BasicRNN/rnn1_vocab.pth',
                            map_location='cuda')
    rnn1 = RNN1().to('cuda')
    rnn1.load_state_dict(state_dict)

    rnn1.eval()
    test_language_model(rnn1)
    sample(rnn1)


def rnn2():
    rnn2 = train_rnn2()

    # Or load the models
    # state_dict = torch.load('dldemos/BasicRNN/rnn2.pth', map_location='cuda')
    # rnn2 = RNN2().to('cuda')
    # rnn2.load_state_dict(state_dict)

    rnn2.eval()
    test_language_model(rnn2, False)
    sample(rnn2)


def main():
    rnn1()
    # rnn2()


if __name__ == '__main__':
    main()
