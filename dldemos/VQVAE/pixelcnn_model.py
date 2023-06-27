import torch.nn as nn

from dldemos.pixelcnn.model import GatedBlock, GatedPixelCNN


class PixelCNNWithEmbedding(GatedPixelCNN):

    def __init__(self, n_blocks, p, linear_dim, bn=True, color_level=256):
        super().__init__(n_blocks, p, linear_dim, bn, color_level)
        self.embedding = nn.Embedding(color_level, p)
        self.block1 = GatedBlock('A', p, p, bn)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return super().forward(x)
