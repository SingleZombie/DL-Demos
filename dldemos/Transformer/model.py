from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        i_seq = torch.linspace(1, max_seq_len, max_seq_len)
        j_seq = torch.linspace(2, d_model, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin((pos / 10000)**(two_i / d_model))
        pe_2i_1 = torch.cos((pos / 10000)**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(1, max_seq_len, d_model)

        self.register_buffer('pe', pe, False)

    def forward(self, x: torch.Tensor):
        n, seq_len, d_model = x.shape
        pe: torch.Tensor = self.pe
        assert seq_len <= pe.shape[0]
        assert d_model == pe.shape[1]
        x *= d_model**0.5
        return x + pe[:, 0:seq_len, :]


def attention(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              mask_len: Optional[int] = None):
    assert q.shape[-1] == k.shape[-1]
    d_k = k.shape[-1]
    tmp = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(d_k)
    if mask_len is not None:
        tmp[..., mask_len:, :] = -torch.inf
    tmp = F.softmax(tmp, -1)
    tmp = torch.matmul(tmp, v)
    return tmp


class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask_len: Optional[int] = None):
        n, seq_len = q.shape[0:2]
        q = self.q(q).reshape(n, seq_len, self.heads, self.d_k).transpose(1, 2)
        k = self.k(k).reshape(n, seq_len, self.heads, self.d_k).transpose(1, 2)
        v = self.v(v).reshape(n, seq_len, self.heads, self.d_k).transpose(1, 2)

        attention_res = attention(q, k, v, mask_len)
        concat_res = attention_res.transpose(1, 2).reshape(
            n, seq_len, self.d_model)
        concat_res = self.dropout(concat_res)

        output = self.out(concat_res)
        return output


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(F.relu(x))
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):

    def __init__(self,
                 heads: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(heads, d_model, dropout)
        self.attention = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_kv: torch.Tensor, mask_len: int):
        x = self.norm1(x +
                       self.dropout(self.self_attention(x, x, x, mask_len)))
        x = self.norm2(x +
                       self.dropout(self.attention(x, encoder_kv, encoder_kv)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(EncoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = self.layer(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(DecoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.Sequential(*self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_kv, mask_len):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = self.layer(x, encoder_kv, mask_len)
        return x


class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 dst_vocab_size: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 80):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_ff, n_layers, heads,
                               dropout, max_seq_len)
        self.decoder = Decoder(src_vocab_size, d_model, d_ff, n_layers, heads,
                               dropout, max_seq_len)

    def forward(self, x, y, mask_len):
        encoder_kv = self.encoder(Encoder)
        res = self.decoder(y, encoder_kv, mask_len)
        embedding_reverse = self.encoder.embedding.weight.transpose(0, 1)
        res = torch.matmul(res, embedding_reverse)
        return res
