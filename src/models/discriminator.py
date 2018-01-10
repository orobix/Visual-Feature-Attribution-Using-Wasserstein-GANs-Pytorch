# encoding: utf8
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    '''TODO: docstring for Discriminator'''

    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.c1 = nn.Conv2d(nc, ndf, 3, 1, 1, bias=False)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.c2 = nn.Conv2d(ndf, 2*ndf, 3, 1, 1, bias=False)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.c3a = nn.Conv2d(2*ndf, 4*ndf, 3, 1, 1, bias=False)
        self.c3b = nn.Conv2d(4*ndf, 4*ndf, 3, 1, 1, bias=False)
        self.pool3 = nn.AvgPool2d(2, 2)

        self.c4a = nn.Conv2d(4*ndf, 8*ndf, 3, 1, 1, bias=False)
        self.c4b = nn.Conv2d(8*ndf, 8*ndf, 3, 1, 1, bias=False)
        self.pool4 = nn.AvgPool2d(2, 2)

        self.c5a = nn.Conv2d(8*ndf, 16*ndf, 3, 1, 1, bias=False)
        self.c5b = nn.Conv2d(16*ndf, 16*ndf, 3, 1, 1, bias=False)
        self.c53 = nn.Conv2d(16*ndf, 16*ndf, 3, 1, 1, bias=False)
        self.c54 = nn.Conv2d(16*ndf, 1, 1, 1, 1)

    def forward(self, x):
        x = F.leaky_relu(self.c1(x), negative_slope=0.2, inplace=True)
        x = self.pool1(x)
        x = F.leaky_relu(self.c2(x), negative_slope=0.2, inplace=True)
        x = self.pool2(x)
        x = F.leaky_relu(self.c3a(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.c3b(x), negative_slope=0.2, inplace=True)
        x = self.pool3(x)
        x = F.leaky_relu(self.c4a(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.c4b(x), negative_slope=0.2, inplace=True)
        x = self.pool4(x)
        x = F.leaky_relu(self.c5a(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.c5b(x), negative_slope=0.2, inplace=True)
        x = F.leaky_relu(self.c53(x), negative_slope=0.2, inplace=True)
        x = F.adaptive_avg_pool2d(F.relu(self.c54(x)), (1, 1))
        return x.mean(0).view(1)
