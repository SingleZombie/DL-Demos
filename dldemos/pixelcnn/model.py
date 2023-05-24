import torch
import torch.nn as nn
import torch.nn.functional as F

# Pixel CNN


class MaskConv2d(nn.Module):

    def __init__(self, conv_type, *args, **kwags):
        super().__init__()
        assert conv_type in ('A', 'B')
        self.conv = nn.Conv2d(*args, **kwags)
        H, W = self.conv.weight.shape[-2:]
        mask = torch.zeros((H, W), dtype=torch.float32)
        mask[0:H // 2] = 1
        mask[H // 2, 0:W // 2] = 1
        if conv_type == 'B':
            mask[H // 2, W // 2] = 1
        mask = mask.reshape((1, 1, H, W))
        self.register_buffer('mask', mask, False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res


class ResidualBlock(nn.Module):

    def __init__(self, h, bn=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2 * h, h, 1)
        self.bn1 = nn.BatchNorm2d(h) if bn else nn.Identity()
        self.conv2 = MaskConv2d('B', h, h, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(h) if bn else nn.Identity()
        self.conv3 = nn.Conv2d(h, 2 * h, 1)
        self.bn3 = nn.BatchNorm2d(2 * h) if bn else nn.Identity()

    def forward(self, x):
        y = self.relu(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        y = y + x
        return y


class PixelCNN(nn.Module):

    def __init__(self, n_blocks, h, linear_dim, bn=True, color_level=256):
        super().__init__()
        self.conv1 = MaskConv2d('A', 1, 2 * h, 7, 1, 3)
        self.bn1 = nn.BatchNorm2d(2 * h) if bn else nn.Identity()
        self.residual_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.residual_blocks.append(ResidualBlock(h, bn))
        self.relu = nn.ReLU()
        self.linear1 = nn.Conv2d(2 * h, linear_dim, 1)
        self.linear2 = nn.Conv2d(linear_dim, linear_dim, 1)
        self.out = nn.Conv2d(linear_dim, color_level, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.relu(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.out(x)
        return x


# Gated PixelCNN


class VerticalMaskConv2d(nn.Module):

    def __init__(self, *args, **kwags):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwags)
        H, W = self.conv.weight.shape[-2:]
        mask = torch.zeros((H, W), dtype=torch.float32)
        mask[0:H // 2 + 1] = 1
        mask = mask.reshape((1, 1, H, W))
        self.register_buffer('mask', mask, False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res


class HorizontalMaskConv2d(nn.Module):

    def __init__(self, conv_type, *args, **kwags):
        super().__init__()
        assert conv_type in ('A', 'B')
        self.conv = nn.Conv2d(*args, **kwags)
        H, W = self.conv.weight.shape[-2:]
        mask = torch.zeros((H, W), dtype=torch.float32)
        mask[H // 2, 0:W // 2] = 1
        if conv_type == 'B':
            mask[H // 2, W // 2] = 1
        mask = mask.reshape((1, 1, H, W))
        self.register_buffer('mask', mask, False)

    def forward(self, x):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)
        return conv_res


class GatedBlock(nn.Module):

    def __init__(self, conv_type, in_channels, p, bn=True):
        super().__init__()
        self.conv_type = conv_type
        self.p = p
        self.v_conv = VerticalMaskConv2d(in_channels, 2 * p, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(2 * p) if bn else nn.Identity()
        self.v_to_h_conv = nn.Conv2d(2 * p, 2 * p, 1)
        self.bn2 = nn.BatchNorm2d(2 * p) if bn else nn.Identity()
        self.h_conv = HorizontalMaskConv2d(conv_type, in_channels, 2 * p, 3, 1,
                                           1)
        self.bn3 = nn.BatchNorm2d(2 * p) if bn else nn.Identity()
        self.h_output_conv = nn.Conv2d(p, p, 1)
        self.bn4 = nn.BatchNorm2d(p) if bn else nn.Identity()

    def forward(self, v_input, h_input):
        v = self.v_conv(v_input)
        v = self.bn1(v)
        v_to_h = v[:, :, 0:-1]
        v_to_h = F.pad(v_to_h, (0, 0, 1, 0))
        v_to_h = self.v_to_h_conv(v_to_h)
        v_to_h = self.bn2(v_to_h)

        v1, v2 = v[:, :self.p], v[:, self.p:]
        v1 = torch.tanh(v1)
        v2 = torch.sigmoid(v2)
        v = v1 * v2

        h = self.h_conv(h_input)
        h = self.bn3(h)
        h = h + v_to_h
        h1, h2 = h[:, :self.p], h[:, self.p:]
        h1 = torch.tanh(h1)
        h2 = torch.sigmoid(h2)
        h = h1 * h2
        h = self.h_output_conv(h)
        h = self.bn4(h)
        if self.conv_type == 'B':
            h = h + h_input
        return v, h


class GatedPixelCNN(nn.Module):

    def __init__(self, n_blocks, p, linear_dim, bn=True, color_level=256):
        super().__init__()
        self.block1 = GatedBlock('A', 1, p, bn)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(GatedBlock('B', p, p, bn))
        self.relu = nn.ReLU()
        self.linear1 = nn.Conv2d(p, linear_dim, 1)
        self.linear2 = nn.Conv2d(linear_dim, linear_dim, 1)
        self.out = nn.Conv2d(linear_dim, color_level, 1)

    def forward(self, x):
        v, h = self.block1(x, x)
        for block in self.blocks:
            v, h = block(v, h)
        x = self.relu(h)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.out(x)
        return x
