import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dldemos.BasicRNN.constant import EMBEDDING_LENGTH, LETTER_LIST, LETTER_MAP


class RNN1(nn.Module):

    def __init__(self, hidden_units=32):
        super().__init__()
        self.hidden_units = hidden_units
        self.linear_a = nn.Linear(hidden_units + EMBEDDING_LENGTH,
                                  hidden_units)
        self.linear_y = nn.Linear(hidden_units, EMBEDDING_LENGTH)
        self.tanh = nn.Tanh()

    def forward(self, word: torch.Tensor):
        # word shape: [batch, max_word_length, embedding_length]
        batch, Tx = word.shape[0:2]

        # word shape: [max_word_length, batch,  embedding_length]
        word = torch.transpose(word, 0, 1)

        # output shape: [max_word_length, batch,  embedding_length]
        output = torch.empty_like(word)

        a = torch.zeros(batch, self.hidden_units, device=word.device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device)
        for i in range(Tx):
            next_a = self.tanh(self.linear_a(torch.cat((a, x), 1)))
            hat_y = self.linear_y(next_a)
            output[i] = hat_y
            x = word[i]
            a = next_a

        # output shape: [batch, max_word_length, embedding_length]
        return torch.transpose(output, 0, 1)

    @torch.no_grad()
    def language_model(self, word: torch.Tensor):
        # word shape: [batch, max_word_length, embedding_length]
        batch, Tx = word.shape[0:2]

        # word shape: [max_word_length, batch,  embedding_length]
        # word_label shape: [max_word_length, batch]
        word = torch.transpose(word, 0, 1)
        word_label = torch.argmax(word, 2)

        # output shape: [batch]
        output = torch.ones(batch, device=word.device)

        a = torch.zeros(batch, self.hidden_units, device=word.device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=word.device)
        for i in range(Tx):
            next_a = self.tanh(self.linear_a(torch.cat((a, x), 1)))
            tmp = self.linear_y(next_a)
            hat_y = F.softmax(tmp, 1)
            probs = hat_y[torch.arange(batch), word_label[i]]
            output *= probs
            x = word[i]
            a = next_a

        return output

    @torch.no_grad()
    def sample_word(self, device='cuda:0'):
        batch = 1
        output = ''

        a = torch.zeros(batch, self.hidden_units, device=device)
        x = torch.zeros(batch, EMBEDDING_LENGTH, device=device)
        for i in range(10):
            next_a = self.tanh(self.linear_a(torch.cat((a, x), 1)))
            tmp = self.linear_y(next_a)
            hat_y = F.softmax(tmp, 1)

            np_prob = hat_y[0].detach().cpu().numpy()
            letter = np.random.choice(LETTER_LIST, p=np_prob)
            output += letter

            if letter == ' ':
                break

            x = torch.zeros(batch, EMBEDDING_LENGTH, device=device)
            x[0][LETTER_MAP[letter]] = 1
            a = next_a

        return output


class RNN2(torch.nn.Module):

    def __init__(self, hidden_units=64, embeding_dim=64, dropout_rate=0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(EMBEDDING_LENGTH, embeding_dim)
        self.rnn = nn.GRU(embeding_dim, hidden_units, 1, batch_first=True)
        self.decoder = torch.nn.Linear(hidden_units, EMBEDDING_LENGTH)
        self.hidden_units = hidden_units

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, word: torch.Tensor):
        # word shape: [batch, max_word_length]
        batch, Tx = word.shape[0:2]
        first_letter = word.new_zeros(batch, 1)
        x = torch.cat((first_letter, word[:, 0:-1]), 1)
        hidden = torch.zeros(1, batch, self.hidden_units, device=word.device)
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        y = self.decoder(output.reshape(batch * Tx, -1))

        return y.reshape(batch, Tx, -1)

    @torch.no_grad()
    def language_model(self, word: torch.Tensor):
        batch, Tx = word.shape[0:2]
        hat_y = self.forward(word)
        hat_y = F.softmax(hat_y, 2)
        output = torch.ones(batch, device=word.device)
        for i in range(Tx):
            probs = hat_y[torch.arange(batch), i, word[:, i]]
            output *= probs

        return output

    @torch.no_grad()
    def sample_word(self, device='cuda:0'):
        batch = 1
        output = ''

        hidden = torch.zeros(1, batch, self.hidden_units, device=device)
        x = torch.zeros(batch, 1, device=device, dtype=torch.long)
        for _ in range(10):
            emb = self.drop(self.encoder(x))
            rnn_output, hidden = self.rnn(emb, hidden)
            hat_y = self.decoder(rnn_output)
            hat_y = F.softmax(hat_y, 2)

            np_prob = hat_y[0, 0].detach().cpu().numpy()
            letter = np.random.choice(LETTER_LIST, p=np_prob)
            output += letter

            if letter == ' ':
                break

            x = torch.zeros(batch, 1, device=device, dtype=torch.long)
            x[0] = LETTER_MAP[letter]

        return output
