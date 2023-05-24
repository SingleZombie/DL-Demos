import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe

from dldemos.SentimentAnalysis.read_imdb import read_imdb

GLOVE_DIM = 100
GLOVE = GloVe(name='6B', dim=GLOVE_DIM)


class IMDBDataset(Dataset):

    def __init__(self, is_train=True, dir='data/aclImdb'):
        super().__init__()
        self.tokenizer = get_tokenizer('basic_english')
        pos_lines = read_imdb(dir, 'pos', is_train)
        neg_lines = read_imdb(dir, 'neg', is_train)
        self.lines = pos_lines + neg_lines
        self.pos_length = len(pos_lines)
        self.neg_length = len(neg_lines)

    def __len__(self):
        return self.pos_length + self.neg_length

    def __getitem__(self, index):
        sentence = self.tokenizer(self.lines[index])
        x = GLOVE.get_vecs_by_tokens(sentence)
        label = 1 if index < self.pos_length else 0
        return x, label


def get_dataloader(dir='data/aclImdb'):

    def collate_fn(batch):
        x, y = zip(*batch)
        x_pad = pad_sequence(x, batch_first=True)
        y = torch.Tensor(y)
        return x_pad, y

    train_dataloader = DataLoader(IMDBDataset(True, dir),
                                  batch_size=32,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(IMDBDataset(False, dir),
                                 batch_size=32,
                                 shuffle=True,
                                 collate_fn=collate_fn)
    return train_dataloader, test_dataloader


class RNN(torch.nn.Module):

    def __init__(self, hidden_units=64, dropout_rate=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.rnn = nn.GRU(GLOVE_DIM, hidden_units, 1, batch_first=True)
        self.linear = nn.Linear(hidden_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # x shape: [batch, max_word_length, embedding_length]
        emb = self.drop(x)
        output, _ = self.rnn(emb)
        output = output[:, -1]
        output = self.linear(output)
        output = self.sigmoid(output)

        return output


def main():
    device = 'cuda:0'
    train_dataloader, test_dataloader = get_dataloader()
    model = RNN().to(device)

    # train

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.BCELoss()
    for epoch in range(100):

        loss_sum = 0
        dataset_len = len(train_dataloader.dataset)

        for x, y in train_dataloader:
            batchsize = y.shape[0]
            x = x.to(device)
            y = y.to(device)
            hat_y = model(x)
            hat_y = hat_y.squeeze(-1)
            loss = citerion(hat_y, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss * batchsize

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

    torch.save(model.state_dict(), 'dldemos/SentimentAnalysis/rnn.pth')

    # test

    # model.load_state_dict(
    #     torch.load('dldemos/SentimentAnalysis/rnn.pth', 'cuda:0'))

    accuracy = 0
    dataset_len = len(test_dataloader.dataset)
    model.eval()
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            hat_y = model(x)
        hat_y.squeeze_(1)
        predictions = torch.where(hat_y > 0.5, 1, 0)
        score = torch.sum(torch.where(predictions == y, 1, 0))
        accuracy += score.item()
    accuracy /= dataset_len

    print(f'Accuracy: {accuracy}')

    # Inference
    tokenizer = get_tokenizer('basic_english')
    article = 'U.S. stock indexes fell Tuesday, driven by expectations for ' \
        'tighter Federal Reserve policy and an energy crisis in Europe. ' \
        'Stocks around the globe have come under pressure in recent weeks ' \
        'as worries about tighter monetary policy in the U.S. and a '\
        'darkening economic outlook in Europe have led investors to '\
        'sell riskier assets.'

    x = GLOVE.get_vecs_by_tokens(tokenizer(article)).unsqueeze(0).to(device)
    with torch.no_grad():
        hat_y = model(x)
    hat_y = hat_y.squeeze_().item()
    result = 'positive' if hat_y > 0.5 else 'negative'
    print(result)


if __name__ == '__main__':
    main()
