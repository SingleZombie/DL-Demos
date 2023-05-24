import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from dldemos.attention.dataset import generate_date, load_date_data

EMBEDDING_LENGTH = 128
OUTPUT_LENGTH = 10


def stoi(str):
    return torch.LongTensor([ord(char) for char in str])


def itos(arr):
    return ''.join([chr(x) for x in arr])


class DateDataset(Dataset):

    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]

        return stoi(line[0]), stoi(line[1])


def get_dataloader(filename):

    def collate_fn(batch):
        x, y = zip(*batch)
        x_pad = pad_sequence(x, batch_first=True)
        y_pad = pad_sequence(y, batch_first=True)
        return x_pad, y_pad

    lines = load_date_data(filename)
    dataset = DateDataset(lines)
    return DataLoader(dataset, 32, collate_fn=collate_fn)


class AttentionModel(nn.Module):

    def __init__(self,
                 embeding_dim=32,
                 encoder_dim=32,
                 decoder_dim=32,
                 dropout_rate=0.5):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(EMBEDDING_LENGTH, embeding_dim)
        self.attention_linear = nn.Linear(2 * encoder_dim + decoder_dim, 1)
        self.softmax = nn.Softmax(-1)
        self.encoder = nn.LSTM(embeding_dim,
                               encoder_dim,
                               1,
                               batch_first=True,
                               bidirectional=True)
        self.decoder = nn.LSTM(EMBEDDING_LENGTH + 2 * encoder_dim,
                               decoder_dim,
                               1,
                               batch_first=True)
        self.output_linear = nn.Linear(decoder_dim, EMBEDDING_LENGTH)
        self.decoder_dim = decoder_dim

    def forward(self, x: torch.Tensor, n_output: int = OUTPUT_LENGTH):
        # x: [batch, n_sequence, EMBEDDING_LENGTH]
        batch, n_squence = x.shape[0:2]

        # x: [batch, n_sequence, embeding_dim]
        x = self.drop(self.embedding(x))

        # a: [batch, n_sequence, hidden]
        a, _ = self.encoder(x)

        # prev_s: [batch, n_squence=1, hidden]
        # prev_y: [batch, n_squence=1, EMBEDDING_LENGTH]
        # y: [batch, n_output, EMBEDDING_LENGTH]
        prev_s = x.new_zeros(batch, 1, self.decoder_dim)
        prev_y = x.new_zeros(batch, 1, EMBEDDING_LENGTH)
        y = x.new_empty(batch, n_output, EMBEDDING_LENGTH)
        tmp_states = None
        for i_output in range(n_output):
            # repeat_s: [batch, n_squence, hidden]
            repeat_s = prev_s.repeat(1, n_squence, 1)
            # attention_input: [batch * n_sequence, hidden_s + hidden_a]
            attention_input = torch.cat((repeat_s, a),
                                        2).reshape(batch * n_squence, -1)
            # x: [batch * n_sequence, 1]
            x = self.attention_linear(attention_input)
            # x: [batch, n_sequence]
            x = x.reshape(batch, n_squence)
            alpha = self.softmax(x)
            c = torch.sum(a * alpha.reshape(batch, n_squence, 1), 1)
            c = c.unsqueeze(1)
            decoder_input = torch.cat((prev_y, c), 2)

            if tmp_states is None:
                prev_s, tmp_states = self.decoder(decoder_input)
            else:
                prev_s, tmp_states = self.decoder(decoder_input, tmp_states)

            prev_y = self.output_linear(prev_s)
            y[:, i_output] = prev_y.squeeze(1)
        return y


def main():
    device = 'cuda:0'
    train_dataloader = get_dataloader('dldemos/attention/train.txt')
    test_dataloader = get_dataloader('dldemos/attention/test.txt')

    model = AttentionModel().to(device)

    # Please close or open the codes with #
    # train

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    citerion = torch.nn.CrossEntropyLoss()
    for epoch in range(30):
        loss_sum = 0
        dataset_len = len(train_dataloader.dataset)

        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            hat_y = model(x)
            n, Tx, _ = hat_y.shape
            hat_y = torch.reshape(hat_y, (n * Tx, -1))
            label_y = torch.reshape(y, (n * Tx, ))
            loss = citerion(hat_y, label_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            loss_sum += loss * n

        print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

    torch.save(model.state_dict(), 'dldemos/attention/model.pth')

    # test
    model.load_state_dict(torch.load('dldemos/attention/model.pth'))

    accuracy = 0
    dataset_len = len(test_dataloader.dataset)

    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        hat_y = model(x)
        prediction = torch.argmax(hat_y, 2)
        score = torch.where(torch.sum(prediction - y, -1) == 0, 1, 0)
        accuracy += torch.sum(score)

    print(f'Accuracy: {accuracy / dataset_len}')

    # inference
    for _ in range(5):
        x, y = generate_date()
        origin_x = x
        x = stoi(x).unsqueeze(0).to(device)
        hat_y = model(x)
        hat_y = hat_y.squeeze(0).argmax(1)
        hat_y = itos(hat_y)
        print(f'input: {origin_x}, prediction: {hat_y}, gt: {y}')


if __name__ == '__main__':
    main()
