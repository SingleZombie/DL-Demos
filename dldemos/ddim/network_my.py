import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)


class ResBlock(nn.Module):

    def __init__(self, shape, in_c, out_c):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        if in_c == out_c:
            self.residual_conv = nn.Identity()
        else:
            self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out += self.residual_conv(x)
        out = self.activation(out)
        return out


class SelfAttentionBlock(nn.Module):

    def __init__(self, shape, dim):
        super().__init__()

        self.ln = nn.LayerNorm(shape)
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        n, c, h, w = x.shape

        norm_x = self.ln(x)
        q = self.q(norm_x)
        k = self.k(norm_x)
        v = self.v(norm_x)

        # n c h w -> n h*w c
        q = q.reshape(n, c, h * w)
        q = q.permute(0, 2, 1)
        # n c h w -> n c h*w
        k = k.reshape(n, c, h * w)

        qk = torch.bmm(q, k) / c**0.5
        qk = torch.softmax(qk, -1)
        # Now qk: [n, h*w, h*w]

        qk = qk.permute(0, 2, 1)
        v = v.reshape(n, c, h * w)
        res = torch.bmm(v, qk)
        res = res.reshape(n, c, h, w)
        res = self.out(res)

        return x + res


class UNetLayer(nn.Module):

    def __init__(self, in_channels, out_channels, h, w, with_attn=False):
        super().__init__()
        self.block1 = ResBlock((in_channels, h, w), in_channels, out_channels)
        self.block2 = ResBlock((out_channels, h, w), out_channels,
                               out_channels)
        if with_attn:
            self.attn = SelfAttentionBlock((out_channels, h, w), out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x, t):
        x = self.block1(x)
        x = x + t
        x = self.block2(x)
        x = self.attn(x)
        return x


class UNet(nn.Module):

    def __init__(self,
                 n_steps,
                 img_shape,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 with_attns=False):
        super().__init__()
        C, H, W = img_shape
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers - 1):
            cH //= 2
            cW //= 2
            Hs.append(cH)
            Ws.append(cW)
        if isinstance(with_attns, bool):
            with_attns = [with_attns] * layers

        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_channel = C
        for channel, cH, cW, with_attn in zip(channels[0:-1], Hs[0:-1],
                                              Ws[0:-1], with_attns[0:-1]):
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, channel),
                              nn.Linear(channel, channel)))

            self.encoders.append(
                UNetLayer(prev_channel, channel, cH, cW, with_attn))
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        self.pe_mid = nn.Sequential(nn.Linear(pe_dim, channels[-1]),
                                    nn.Linear(channels[-1], channels[-1]))
        self.mid = UNetLayer(prev_channel, channels[-1], Hs[-1], Ws[-1],
                             with_attns[-1])
        prev_channel = channels[-1]
        for channel, cH, cW, with_attn in zip(channels[-2::-1], Hs[-2::-1],
                                              Ws[-2::-1], with_attns[-2::-1]):
            self.pe_linears_de.append(
                nn.Sequential(nn.Linear(pe_dim, channel),
                              nn.Linear(channel, channel)))
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))
            self.decoders.append(
                UNetLayer(channel * 2, channel, cH, cW, with_attn))

            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        encoder_outs = []
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders,
                                            self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = encoder(x, pe)
            encoder_outs.append(x)
            x = down(x)
        pe = self.pe_mid(t).reshape(n, -1, 1, 1)
        x = self.mid(x, pe)
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = up(x)

            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x, pe)
        x = self.conv_out(x)
        return x
