from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MY_INF = 1e12


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(1, max_seq_len, d_model)

        self.register_buffer('pe', pe, False)

    def forward(self, x: torch.Tensor):
        n, seq_len, d_model = x.shape
        pe: torch.Tensor = self.pe
        assert seq_len <= pe.shape[1]
        assert d_model == pe.shape[2]
        rescaled_x = x * d_model**0.5
        return rescaled_x + pe[:, 0:seq_len, :]


def attention(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              mask: Optional[torch.Tensor] = None):
    '''
    Note: The dtype of mask must be bool
    '''
    # q shape: [n, heads, q_len, d_k]
    # k shape: [n, heads, k_len, d_k]
    # v shape: [n, heads, k_len, d_v]
    assert q.shape[-1] == k.shape[-1]
    d_k = k.shape[-1]
    # tmp shape: [n, heads, q_len, k_len]
    tmp = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5
    if mask is not None:
        tmp.masked_fill_(mask, -MY_INF)
    tmp = F.softmax(tmp, -1)
    # tmp shape: [n, heads, q_len, d_v]
    tmp = torch.matmul(tmp, v)
    return tmp


class MultiHeadAttention(nn.Module):

    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % heads == 0
        # dk == dv
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
                mask: Optional[torch.Tensor] = None):
        # batch should be same
        assert q.shape[0] == k.shape[0]
        assert q.shape[0] == v.shape[0]
        # the sequence length of k and v should be aligned
        assert k.shape[1] == v.shape[1]

        n, q_len = q.shape[0:2]
        n, k_len = k.shape[0:2]
        q_ = self.q(q).reshape(n, q_len, self.heads, self.d_k).transpose(1, 2)
        k_ = self.k(k).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)
        v_ = self.v(v).reshape(n, k_len, self.heads, self.d_k).transpose(1, 2)

        attention_res = attention(q_, k_, v_, mask)
        concat_res = attention_res.transpose(1, 2).reshape(
            n, q_len, self.d_model)
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
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        tmp = self.self_attention(x, x, x, src_mask)
        tmp = self.dropout1(tmp)
        x = self.norm1(x + tmp)
        tmp = self.ffn(x)
        tmp = self.dropout2(tmp)
        x = self.norm2(x + tmp)
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
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                x,
                encoder_kv: torch.Tensor,
                dst_mask: Optional[torch.Tensor] = None,
                src_dst_mask: Optional[torch.Tensor] = None):
        tmp = self.self_attention(x, x, x, dst_mask)
        tmp = self.dropout1(tmp)
        x = self.norm1(x + tmp)
        tmp = self.attention(x, encoder_kv, encoder_kv, src_dst_mask)
        tmp = self.dropout2(tmp)
        x = self.norm2(x + tmp)
        tmp = self.ffn(x)
        tmp = self.dropout3(tmp)
        x = self.norm3(x + tmp)
        return x


class Encoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(EncoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.ModuleList(self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 120):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_seq_len)
        self.layers = []
        for i in range(n_layers):
            self.layers.append(DecoderLayer(heads, d_model, d_ff, dropout))
        self.layers = nn.Sequential(*self.layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x,
                encoder_kv,
                dst_mask: Optional[torch.Tensor] = None,
                src_dst_mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_kv, dst_mask, src_dst_mask)
        return x


class Transformer(nn.Module):

    def __init__(self,
                 src_vocab_size: int,
                 dst_vocab_size: int,
                 pad_idx: int,
                 d_model: int,
                 d_ff: int,
                 n_layers: int,
                 heads: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 200):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, pad_idx, d_model, d_ff,
                               n_layers, heads, dropout, max_seq_len)
        self.decoder = Decoder(dst_vocab_size, pad_idx, d_model, d_ff,
                               n_layers, heads, dropout, max_seq_len)
        self.pad_idx = pad_idx
        self.output_layer = nn.Linear(d_model, dst_vocab_size)

    def generate_mask(self,
                      q_pad: torch.Tensor,
                      k_pad: torch.Tensor,
                      with_left_mask: bool = False):
        # q_pad shape: [n, q_len]
        # k_pad shape: [n, k_len]
        # q_pad k_pad dtype: bool
        assert q_pad.device == k_pad.device
        n, q_len = q_pad.shape
        n, k_len = k_pad.shape

        mask_shape = (n, 1, q_len, k_len)
        if with_left_mask:
            mask = 1 - torch.tril(torch.ones(mask_shape))
        else:
            mask = torch.zeros(mask_shape)
        mask = mask.to(q_pad.device)
        for i in range(n):
            mask[i, :, q_pad[i], :] = 1
            mask[i, :, :, k_pad[i]] = 1
        mask = mask.to(torch.bool)
        return mask

    def forward(self, x, y):

        src_pad_mask = x == self.pad_idx
        dst_pad_mask = y == self.pad_idx
        src_mask = self.generate_mask(src_pad_mask, src_pad_mask, False)
        dst_mask = self.generate_mask(dst_pad_mask, dst_pad_mask, True)
        src_dst_mask = self.generate_mask(dst_pad_mask, src_pad_mask, False)
        encoder_kv = self.encoder(x, src_mask)
        res = self.decoder(y, encoder_kv, dst_mask, src_dst_mask)
        res = self.output_layer(res)
        return res
