import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
import torchvision
import torchvision.transforms as transforms
from PIL import Image

device = 'cuda'


def downsample(x, scale):
    downsample = nn.AvgPool2d(scale, stride=scale)
    return downsample(x)


def upsample(x, scale):
    upsample = nn.Upsample(scale_factor=scale, mode='bilinear')
    return upsample(x)


def single_conv_module(in_chs, out_chs, kernel, deconv=False, activation=True, leaky=True):
    assert kernel % 2 == 1

    layers = [
        nn.Conv2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel-1)//2),
        nn.BatchNorm2d(out_chs),
        nn.LeakyReLU(0.2, inplace=True)
    ]

    if deconv:
        layers[0] = nn.ConvTranspose2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel-1)//2)
    if not activation:
        del layers[-1]
    if not leaky:
        layers[2] = nn.ReLu(inplace=True)

    return nn.Sequential(*layers)


def downsampling_module(in_chs, pooling_kenel, leaky=True):
    layers = [
        nn.AvgPool2d(pooling_kenel),
        nn.Conv2d(in_chs, in_chs*2, kernel_size=1, padding=0),
        nn.BatchNorm2d(in_chs*2),
        nn.LeakyReLU(0.2, inplace=True)
    ]

    if not leaky:
        layers[3] = nn.ReLu(inplace=True)

    return nn.Sequential(*layers)


def upsampling_module(in_chs, pooling_kenel, leaky=True):
    layers = [
        nn.ConvTranspose2d(in_chs, in_chs//2, kernel_size=pooling_kenel, stride=pooling_kenel),
        nn.BatchNorm2d(in_chs//2),
        nn.LeakyReLU(0.2, inplace=True)
    ]

    if not leaky:
        layers[2] = nn.ReLu(inplace=True)

    return nn.Sequential(*layers)


class inception_module(nn.Module):
    def __init__(self, in_chs, out_chs, deconv=False, cat=False, leaky=True):
        super().__init__()
        bn_ch = out_chs // 2

        self.residual = in_chs == out_chs
        self.flag_cat = cat
        self.activation = False

        if self.flag_cat:
            self.bottleneck = single_conv_module(bn_ch*6, out_chs, 1, deconv=deconv)
            out_chs = bn_ch

        self.conv1 = single_conv_module(in_chs, out_chs, 1, deconv=deconv, activation=self.activation, leaky=leaky)

        self.conv3 = nn.Sequential(
            single_conv_module(in_chs, bn_ch, 1, deconv=deconv, leaky=leaky),
            single_conv_module(bn_ch, out_chs, 3, deconv=deconv, activation=self.activation, leaky=leaky)
        )

        self.conv5 = nn.Sequential(
            single_conv_module(in_chs, bn_ch, 1, deconv=deconv, leaky=leaky),
            single_conv_module(bn_ch, bn_ch, 3, deconv=deconv, leaky=leaky),
            single_conv_module(bn_ch, out_chs, 3, deconv=deconv, activation=self.activation, leaky=leaky)
        )

        self.conv7 = nn.Sequential(
            single_conv_module(in_chs, bn_ch, 1, deconv=deconv, leaky=leaky),
            single_conv_module(bn_ch, bn_ch, 3, deconv=deconv, leaky=leaky),
            single_conv_module(bn_ch, bn_ch, 3, deconv=deconv, leaky=leaky),
            single_conv_module(bn_ch, out_chs, 3, deconv=deconv, activation=self.activation, leaky=leaky)
        )

        self.pool3 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            single_conv_module(in_chs, out_chs, 1, deconv=deconv, activation=self.activation, leaky=leaky)
            )

        self.pool5 = nn.Sequential(
            nn.MaxPool2d(5, stride=1, padding=2),
            single_conv_module(in_chs, out_chs, 1, deconv=deconv, activation=self.activation, leaky=leaky)
            )

        self.final_activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if not self.flag_cat:
            y = self.conv1(x) + self.conv3(x) + self.conv5(x) + self.conv7(x) + self.pool3(x) + self.pool5(x)
            if self.residual:
                y = y + x
            if not self.activation:
                y = self.final_activation(y)
        else:
            y = torch.cat((self.conv1(x), self.conv3(x), self.conv5(x), self.conv7(x), self.pool3(x), self.pool5(x)), 1)
            y = self.bottleneck[0](y)
            y = self.bottleneck[1](y)
            if self.residual:
                y = y + x
            y = self.bottleneck[2](y)
        
        return y


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, scale):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            inception_module(64, 64, leaky=True)
        )
        self.stage2 = nn.Sequential(
            downsampling_module(64, 2),
            inception_module(128, 128, leaky=True)
        )
        self.stage3 = nn.Sequential(
            downsampling_module(128, 2),
            inception_module(256, 256, leaky=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.convs_mu = nn.Sequential(nn.Conv2d(256, latent_dim, kernel_size=1, padding=0))
        self.convs_logvar = nn.Sequential(nn.Conv2d(256, latent_dim, kernel_size=1, padding=0))

        self.H = int(48/scale)
        self.W = int(64/scale)

    def forward(self, x):                   # [N, 3, 48, 64]
        self.f1 = self.stage1(x)            # [N, 64, 48, 64]
        self.f2 = self.stage2(self.f1)      # [N, 128, 24, 32]
        self.f3 = self.stage3(self.f2)      # [N, 256, 12, 16]
        self.features = [self.f1, self.f2, self.f3]

        mu = self.convs_mu(self.f3)         # [N, latent_dim, 12, 16]
        logvar = self.convs_logvar(self.f3) # [N, latent_dim, 12, 16]
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_dim, z_S_ch, scale):
        super().__init__()
        self.H = int(48/scale)
        self.W = int(64/scale)

        self.inconv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + z_S_ch, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.stage1 = inception_module(256, 256, deconv=True, leaky=True)
        self.stage2 = nn.Sequential(
            upsampling_module(256, 2),
            inception_module(128, 128, deconv=True, leaky=True)
        )
        self.stage3 = nn.Sequential(
            upsampling_module(128, 2),
            inception_module(64, 64, deconv=True, leaky=True)
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels=out_channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, z_I, z_S):
        z = torch.cat((z_I, z_S), 1)        # [N, latent_dim + z_S_ch, 12, 16]
        y = self.inconv(z)                  # [N, 256, 12, 16]

        self.f3 = self.stage1(y)            # [N, 256, 12, 16]
        self.f2 = self.stage2(self.f3)      # [N, 128, 24, 32]
        self.f1 = self.stage3(self.f2)      # [N, 64, 48, 64]
        self.features = [self.f1, self.f2, self.f3]
        y = self.last(self.f1)
        return y                            # [N, 3, 48, 64]


class Discriminator(nn.Module):
    def __init__(self, in_channels, scale):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            inception_module(32, 32, cat=False)
        )
        self.stage2 = nn.Sequential(
            downsampling_module(32, 2),
            inception_module(64, 64, cat=False)
        )
        self.stage3 = nn.Sequential(
            downsampling_module(64, 2),
            inception_module(128, 128, cat=False)
        )

        self.final = nn.Sequential(
            nn.Conv2d(129, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):                   # [N, 3, 48, 64]
        self.f1 = self.stage1(x)            # [N, 32, 48, 64]
        self.f2 = self.stage2(self.f1)      # [N, 64, 24, 32]
        self.f3 = self.stage3(self.f2)      # [N, 128, 12, 16]
        self.features = [self.f1, self.f2, self.f3]
        x = self.features[-1]
        
        msd = self.minibatch_stddev(x).view(1,1,1,1)
        msd = msd.expand(x.size(0), 1, x.size(2), x.size(3))
        y = torch.cat((x, msd), 1)

        y = self.final(y)
        return y

    def minibatch_stddev(self, x):      #[NCHW]
        stddev = torch.std(x, dim=0)    #[CHW]
        stddev = torch.mean(stddev)     #scalar
        return stddev


class MapGenerator(nn.Module):
    def __init__(self, z_S_ch, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            inception_module(64, 64),
            downsampling_module(64, 2),

            inception_module(128, 128),
            downsampling_module(128, 2),

            inception_module(256, 256),

            single_conv_module(256, z_S_ch, 1)
        )

        self.decoder = nn.Sequential(
            single_conv_module(z_S_ch, 256, 1, deconv=True),

            inception_module(256, 256, deconv=True),
            upsampling_module(256, 2),

            inception_module(128, 128, deconv=True),
            upsampling_module(128, 2),

            inception_module(64, 64, deconv=True),

            nn.ConvTranspose2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        self.z_S = self.encoder(x)
        y = self.decoder(self.z_S)
        return y