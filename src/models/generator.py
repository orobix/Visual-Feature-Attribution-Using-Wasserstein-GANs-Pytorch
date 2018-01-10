# encoding: utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """docstring for Generator"""

    def __init__(self, nc=1, ndf=16):
        super(Generator, self).__init__()
        # encoder
        self.c1a = nn.Conv2d(nc, ndf, 3, 1, 1, bias=False)
        self.bn1a = nn.BatchNorm2d(ndf)
        self.c1b = nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False)
        self.bn1b = nn.BatchNorm2d(ndf)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.c2a = nn.Conv2d(ndf, 2*ndf, 3, 1, 1, bias=False)
        self.bn2a = nn.BatchNorm2d(2*ndf)
        self.c2b = nn.Conv2d(2*ndf, 2*ndf, 3, 1, 1, bias=False)
        self.bn2b = nn.BatchNorm2d(2*ndf)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.c3a = nn.Conv2d(2*ndf, 4*ndf, 3, 1, 2, bias=False)
        self.bn3a = nn.BatchNorm2d(4*ndf)
        self.c3b = nn.Conv2d(4*ndf, 4*ndf, 3, 1, 2, bias=False)
        self.bn3b = nn.BatchNorm2d(4*ndf)
        self.pool3 = nn.AvgPool2d(2, 2)

        # bottleneck
        self.c4a = nn.Conv2d(4*ndf, 8*ndf, 3, 1, 1, bias=False)
        self.bn4a = nn.BatchNorm2d(8*ndf)
        self.c4b = nn.Conv2d(8*ndf, 8*ndf, 3, 1, 1, bias=False)
        self.bn4b = nn.BatchNorm2d(8*ndf)

        # decoder
        self.uc3 = nn.ConvTranspose2d(8*ndf, 4*ndf, 4, 2, 0, bias=False)
        self.ubn3 = nn.BatchNorm2d(4*ndf)
        self.c5a = nn.Conv2d(8*ndf, 4*ndf, 3, 1, 0, bias=False)
        self.bn5a = nn.BatchNorm2d(4*ndf)
        self.c5b = nn.Conv2d(4*ndf, 4*ndf, 3, 1, 0, bias=False)
        self.bn5b = nn.BatchNorm2d(4*ndf)

        self.uc2 = nn.ConvTranspose2d(4*ndf, 2*ndf, 4, 2, 0, bias=False)
        self.ubn2 = nn.BatchNorm2d(2*ndf)
        self.c6a = nn.Conv2d(4*ndf, 2*ndf, 3, 1, 1, bias=False)
        self.bn6a = nn.BatchNorm2d(2*ndf)
        self.c6b = nn.Conv2d(2*ndf, 2*ndf, 3, 1, 1, bias=False)
        self.bn6b = nn.BatchNorm2d(2*ndf)

        self.uc1 = nn.ConvTranspose2d(2*ndf, ndf, 4, 2, 0, bias=False)
        self.ubn1 = nn.BatchNorm2d(ndf)
        self.c8a = nn.Conv2d(2*ndf, ndf, 3, 1, 1, bias=False)
        self.bn8a = nn.BatchNorm2d(ndf)
        self.c8b = nn.Conv2d(ndf, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        x1a = F.leaky_relu(self.bn1a(self.c1a(x)),
                           negative_slope=0.2, inplace=True)
        x1b = F.leaky_relu(self.bn1b(self.c1b(x1a)),
                           negative_slope=0.2, inplace=True)
        x1 = self.pool1(x1b)  # 16, 56, 56

        x2a = F.leaky_relu(self.bn2a(self.c2a(x1)),
                           negative_slope=0.2, inplace=True)
        x2b = F.leaky_relu(self.bn2b(self.c2b(x2a)),
                           negative_slope=0.2, inplace=True)
        x2 = self.pool2(x2b)  # 32, 28, 28

        x3a = F.leaky_relu(self.bn3a(self.c3a(x2)),
                           negative_slope=0.2, inplace=True)
        x3b = F.leaky_relu(self.bn3b(self.c3b(x3a)),
                           negative_slope=0.2, inplace=True)
        x3 = self.pool3(x3b)  # 32, 16, 16

        x4a = F.leaky_relu(self.bn4a(self.c4a(x3)),
                           negative_slope=0.2, inplace=True)
        x4b = F.leaky_relu(self.bn4b(self.c4b(x4a)),
                           negative_slope=0.2, inplace=True)  # 128, 16, 16

        xu3 = F.leaky_relu(self.ubn3(self.uc3(x4b)),
                           negative_slope=0.2, inplace=True)

        xu3b = torch.cat([xu3[:, :, 1:-1, 1:-1], x3b], dim=1)
        x5a = F.leaky_relu(self.bn5a(self.c5a(xu3b)),
                           negative_slope=0.2, inplace=True)  # 64, 30, 30
        x5b = F.leaky_relu(self.bn5b(self.c5b(x5a)),
                           negative_slope=0.2, inplace=True)  # 64, 28, 28

        xu2 = F.leaky_relu(self.ubn2(self.uc2(x5b)),
                           negative_slope=0.2, inplace=True)  # 64, 58, 58
        xu2b = torch.cat([xu2[:, :, 1:-1, 1:-1], x2b], dim=1)  # 64, 56, 56
        x6a = F.leaky_relu(self.bn6a(self.c6a(xu2b)),
                           negative_slope=0.2, inplace=True)  # 32, 56, 56
        x6b = F.leaky_relu(self.bn6b(self.c6b(x6a)),
                           negative_slope=0.2, inplace=True)

        xu1 = F.leaky_relu(self.ubn1(self.uc1(x6b)),
                           negative_slope=0.2, inplace=True)
        xu1b = torch.cat([xu1[:, :, 1:-1, 1:-1], x1b], dim=1)
        x8a = F.leaky_relu(self.bn8a(self.c8a(xu1b)),
                           negative_slope=0.2, inplace=True)
        x8 = self.c8b(x8a)
        return F.tanh(x8)
