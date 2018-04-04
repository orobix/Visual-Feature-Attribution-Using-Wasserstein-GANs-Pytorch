import torch.nn as nn
from models.model_utils import conv2d_bn_block, dense_layer_bn, Identity


class NormalNet2D(nn.Module):
    '''
    portings for the models found in the reference reference's official repo
    https://github.com/baumgach/vagan-code
    '''
    def __init__(self, n_channels=1, nlabels=1, init_filters=32):
        nf = init_filters
        super(NormalNet2D, self).__init__()
        self.encoder = nn.Sequential(
            conv2d_bn_block(n_channels, nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(nf, 2*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(2*nf, 4*nf),
            conv2d_bn_block(4*nf, 4*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(4*nf, 8*nf),
            conv2d_bn_block(8*nf, 8*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(8*nf, 16*nf),
            conv2d_bn_block(16*nf, 16*nf),
            conv2d_bn_block(16*nf, 16*nf),
        )
        self.classifier = nn.Sequential(
            dense_layer_bn(16*nf, 16*nf),
            dense_layer_bn(16*nf, nlabels, activation=Identity)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class RebuttalNet2D(nn.Module):
    '''
    portings for the models found in the reference reference's official repo
    https://github.com/baumgach/vagan-code
    '''
    def __init__(self, n_channels=1, nlabels=1, init_filters=32):
        nf = init_filters
        super(RebuttalNet2D, self).__init__()
        self.encoder = nn.Sequential(
            conv2d_bn_block(n_channels, nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(nf, 2*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(2*nf, 4*nf),
            conv2d_bn_block(4*nf, 4*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(4*nf, 8*nf),
            conv2d_bn_block(8*nf, 8*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(8*nf, 16*nf),
            conv2d_bn_block(16*nf, 16*nf),
            conv2d_bn_block(16*nf, 16*nf),
        )
        self.classifier = nn.Sequential(
            nn.AvgPool2d(2),
            dense_layer_bn(16*nf, nlabels, activation=Identity)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class NormalNetDeeper2D(nn.Module):
    '''
    portings for the models found in the reference reference's official repo
    https://github.com/baumgach/vagan-code
    '''
    def __init__(self, n_channels=1, nlabels=1, init_filters=32):
        nf = init_filters
        super(NormalNetDeeper2D, self).__init__()
        self.encoder = nn.Sequential(
            conv2d_bn_block(n_channels, nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(nf, 2*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(2*nf, 4*nf),
            conv2d_bn_block(4*nf, 4*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(4*nf, 8*nf),
            conv2d_bn_block(8*nf, 8*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(8*nf, 16*nf),
            conv2d_bn_block(16*nf, 16*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(16*nf, 16*nf),
            conv2d_bn_block(16*nf, 16*nf),
            conv2d_bn_block(16*nf, 16*nf),
        )
        self.classifier = nn.Sequential(
            dense_layer_bn(16*nf, 16*nf),
            dense_layer_bn(16*nf, nlabels, activation=Identity)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class CamNet2D(nn.Module):
    '''
    portings for the models found in the reference reference's official repo
    https://github.com/baumgach/vagan-code
    '''
    def __init__(self, n_channels=1, nlabels=1, init_filters=32):
        nf = init_filters
        super(CamNet2D, self).__init__()
        self.encoder = nn.Sequential(
            conv2d_bn_block(n_channels, nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(nf, 2*nf),
            nn.MaxPool2d(2),
            conv2d_bn_block(2*nf, 4*nf),
            conv2d_bn_block(4*nf, 4*nf),
            conv2d_bn_block(4*nf, 8*nf),
            conv2d_bn_block(8*nf, 8*nf),
            conv2d_bn_block(8*nf, 16*nf),
            conv2d_bn_block(16*nf, 16*nf),
            conv2d_bn_block(16*nf, 16*nf),
        )
        self.classifier = nn.Sequential(
            nn.AvgPool2d(2),
            dense_layer_bn(16*nf, nlabels, activation=Identity)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
