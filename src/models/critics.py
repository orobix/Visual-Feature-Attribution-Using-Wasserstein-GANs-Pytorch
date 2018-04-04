import torch.nn as nn
from models.model_utils import (conv2d_block,
                                conv2d_bn_block,
                                conv3d_block,
                                Identity)


class C3DFCN(nn.Module):
    """docstring for C3DFCN2D"""
    def __init__(self, n_channels=1, init_filters=16, dimensions=2, batch_norm=False):
        super(C3DFCN, self).__init__()
        nf = init_filters
        if dimensions == 2:
            conv_block = conv2d_bn_block if batch_norm else conv2d_block
        else:
            conv_block = conv3d_block
        max_pool = nn.MaxPool2d if int(dimensions) is 2 else nn.MaxPool3d
        self.encoder = nn.Sequential(
            conv_block(n_channels, nf),
            max_pool(2),
            conv_block(nf, 2*nf),
            max_pool(2),
            conv_block(2*nf, 4*nf),
            conv_block(4*nf, 4*nf),
            max_pool(2),
            conv_block(4*nf, 8*nf),
            conv_block(8*nf, 8*nf),
            max_pool(2),
            conv_block(8*nf, 16*nf),
            conv_block(16*nf, 16*nf),
            conv_block(16*nf, 16*nf),
        )
        self.classifier = nn.Sequential(
            conv_block(16*nf, 1, kernel=1, activation=Identity),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1).mean(1)
