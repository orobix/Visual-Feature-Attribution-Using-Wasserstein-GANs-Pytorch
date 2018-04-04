import torch.nn as nn
from models.model_utils import conv3d_bn_block, dense_layer_bn


class FCNBN(nn.Module):
    '''
    portings for the models found in the reference reference's official repo
    https://github.com/baumgach/vagan-code
    '''
    def __init__(self, n_channels=1, nlabels=1, init_filters=32):
        nf = init_filters
        super(FCNBN, self).__init__()
        self.encoder = nn.Sequential(
            conv3d_bn_block(n_channels, nf),
            nn.MaxPool3d(2),
            conv3d_bn_block(nf, 2*nf),
            nn.MaxPool3d(2),
            conv3d_bn_block(2*nf, 4*nf),
            conv3d_bn_block(4*nf, 4*nf),
            nn.MaxPool3d(2),
            conv3d_bn_block(4*nf, 8*nf),
            conv3d_bn_block(8*nf, 8*nf),
            nn.MaxPool3d(2),
            conv3d_bn_block(8*nf, 8*nf),
            conv3d_bn_block(8*nf, 8*nf),
            conv3d_bn_block(8*nf, 8*nf),
        )
        self.classifier = nn.Sequential(
            conv3d_bn_block(8*nf, nlabels, kernel=1, activation=lambda x: x),
            nn.AvgPool3d(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class AllConvBN(nn.Module):
    '''
    portings for the models found in the reference reference's official repo
    https://github.com/baumgach/vagan-code
    '''
    def __init__(self, n_channels=1, nlabels=1, init_filters=32):
        nf = init_filters
        super(AllConvBN, self).__init__()
        self.encoder = nn.Sequential(
            conv3d_bn_block(n_channels, nf, stride=2),
            conv3d_bn_block(nf, 2*nf, stride=2),
            conv3d_bn_block(2*nf, 4*nf),
            conv3d_bn_block(4*nf, 4*nf, stride=2),
            conv3d_bn_block(4*nf, 8*nf),
            conv3d_bn_block(8*nf, 8*nf, stride=2),
            conv3d_bn_block(8*nf, 8*nf),
            conv3d_bn_block(8*nf, 8*nf),
            conv3d_bn_block(8*nf, 8*nf),
        )
        self.classifier = nn.Sequential(
            conv3d_bn_block(8*nf, nlabels, kernel=1, activation=lambda x: x),
            nn.AvgPool3d(2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


class C3DBN(nn.Module):
    '''
    portings for the models found in the reference reference's official repo
    https://github.com/baumgach/vagan-code
    '''
    def __init__(self, n_channels=1, nlabels=1, init_filters=32):
        nf = init_filters
        super(AllConvBN, self).__init__()
        self.encoder = nn.Sequential(
            conv3d_bn_block(n_channels, nf),
            nn.MaxPool3d(2),
            conv3d_bn_block(nf, 2*nf),
            nn.MaxPool3d(2),
            conv3d_bn_block(2*nf, 4*nf),
            conv3d_bn_block(4*nf, 4*nf),
            nn.MaxPool3d(2),
            conv3d_bn_block(4*nf, 8*nf),
            conv3d_bn_block(8*nf, 8*nf),
            nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            dense_layer_bn(8*nf, 16*nf),
            dense_layer_bn(16*nf, nlabels, activation=lambda x: x),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
