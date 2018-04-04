import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATION = nn.ReLU


def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))
    return torch.cat((upsampled, bypass), 1)


def conv2d_bn_block(in_channels, out_channels, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block conv-bn-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def deconv2d_bn_block(in_channels, out_channels, kernel=4, stride=2, padding=1, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block deconv-bn-activation
    '''
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, momentum=momentum),
        activation(),
    )


def dense_layer_bn(in_dim, out_dim, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block linear-bn-activation
    '''
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim, momentum=momentum),
        activation()
    )


def conv3d_bn_block(in_channels, out_channels, kernel=3, stride=1, padding=1, momentum=0.01, activation=ACTIVATION):
    '''
    returns a block 3Dconv-3Dbn-activation
    '''
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        nn.BatchNorm3d(out_channels, momentum=momentum),
        activation(),
    )


def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    '''
    returns a block conv-activation
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


def conv3d_block(in_channels, out_channels, kernel=3, stride=1, padding=1, activation=ACTIVATION):
    '''
    returns a block 3D conv-activation
    '''
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding),
        activation(),
    )


class Identity(nn.Module):

    def forward(self, x):
        return x
