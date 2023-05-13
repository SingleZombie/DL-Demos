import torch
import torch.nn as nn


class MaskConv2d(nn.Module):

    def __init__(self, conv_type, *args, **kwags):
        super().__init__()
        assert conv_type in ['A', 'B']
        self.conv = nn.Conv2d(*args, **kwags)
        H, W = self.conv.weight.shape[-2:]
        mask = torch.zeros((H, W), dtype=torch.float32)
        mask[0:H // 2] = 1
        mask[H // 2, 0:W // 2] = 1
        if conv_type == 'B':
            mask[H // 2, W // 2] = 1
        self.register_buffer('mask', mask, False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res


class ResidualBlock(nn.Module):

    def __init__(self, h):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2 * h, h, 1)
        self.conv2 = MaskConv2d('B', h, h, 3, 1, 1)
        self.conv3 = nn.Conv2d(h, 2 * h, 1)

    def forward(self, x):
        y = self.relu(x)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = y + x
        return y


class PixelCNN(nn.Module):

    def __init__(self, h, linear_dim, n_class=256):
        super().__init__()
        self.conv1 = MaskConv2d('A', 1, 2 * h, 7, 1, 3)
        self.residual_blocks = nn.ModuleList()
        for _ in range(15):
            self.residual_blocks.append(ResidualBlock(h))
        self.relu = nn.ReLU()
        self.linear1 = nn.Conv2d(2 * h, linear_dim, 1)
        self.linear2 = nn.Conv2d(linear_dim, linear_dim, 1)
        self.out = nn.Conv2d(linear_dim, n_class, 1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.out(x)
        return x