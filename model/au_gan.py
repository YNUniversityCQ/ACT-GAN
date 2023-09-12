
# AW-GAN (Radio map estimate)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from .attention import Spatial_Attention,cbam_block,cbam_block1
from .common import BaseNetwork

def convrelu(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.ReLU(True)
    )

def convrelus(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.ReLU(True)
    )

def convreluss(in_channels, out_channels, kernel, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
    )

class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


def upconvrelu(in_channels, out_channels):
    return nn.Sequential(
        UpConv(in_channels, out_channels),
        nn.ReLU(True)
    )


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        # dilation rate
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))

        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]

        out = torch.cat(out, 1)

        out = self.fuse(out)

        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


# Define Generator

class Generator(BaseNetwork):
    def __init__(self, args):
        super(Generator, self).__init__()

        # PHASE 1
        '''
        Scenario 1 Enter dimensions：2
        Scenario 2 Enter dimensions：3
        Scenario 3 Enter dimensions：2
        '''
        self.conv1 = convrelus(2, 64, 7, 1, 0)  # size:256
        self.att1 = cbam_block(64)
        self.middle1 = nn.Sequential(*[AOTBlock(64, args.rates) for _ in range(args.block_num)])  # channel:64

        self.conv2 = convrelu(64, 128, 4, 2, 1)  # size:128
        self.att2 = cbam_block(128)
        self.middle2 = nn.Sequential(*[AOTBlock(128, args.rates) for _ in range(args.block_num)])  # channel:128

        self.conv3 = convrelu(128, 256, 4, 2, 1)  # size:64
        self.att3 = cbam_block(256)
        self.middle3 = nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])  # channel:256

        self.conv4 = convrelu(256, 512, 4, 2, 1)  # size:32
        self.att4 = cbam_block(512)
        self.middle4 = nn.Sequential(*[AOTBlock(512, args.rates) for _ in range(args.block_num)])  # channel:512

        self.upconv1 = upconvrelu(512, 256)  # size:64
        self.cat1 = convrelu(512, 256, 3, 1, 1)

        self.upconv2 = upconvrelu(256, 128)  # size:128
        self.cat2 = convrelu(256, 128, 3, 1, 1)

        self.upconv3 = upconvrelu(128, 64)  # size:256
        self.cat3 = convrelu(128, 64, 3, 1, 1)

        self.upconv4 = convreluss(64, 1, 3, 1, 1)  # size:256

        # PHASE 2

        self.conv5 = convrelus(1, 64, 7, 1, 0)  # size:256
        self.att5 = cbam_block(64)
        self.middle5 = nn.Sequential(*[AOTBlock(64, args.rates) for _ in range(args.block_num)])  # channel:64

        self.conv6 = convrelu(64, 128, 4, 2, 1)  # size:128
        self.att6 = cbam_block(128)
        self.middle6 = nn.Sequential(*[AOTBlock(128, args.rates) for _ in range(args.block_num)])  # channel:128

        self.conv7 = convrelu(128, 256, 4, 2, 1)  # size:64
        self.att7 = cbam_block(256)
        self.middle7 = nn.Sequential(*[AOTBlock(256, args.rates) for _ in range(args.block_num)])  # channel:256

        self.conv8 = convrelu(256, 512, 4, 2, 1)  # size:32
        self.att8 = cbam_block(512)
        self.middle8 = nn.Sequential(*[AOTBlock(512, args.rates) for _ in range(args.block_num)])  # channel:512

        self.upconv5 = upconvrelu(512, 256)  # size:64
        self.cat4 = convrelu(512, 256, 3, 1, 1)

        self.upconv6 = upconvrelu(256, 128)  # size:128
        self.cat5 = convrelu(256, 128, 3, 1, 1)

        self.upconv7 = upconvrelu(128, 64)  # size:256
        self.cat6 = convrelu(128, 64, 3, 1, 1)

        self.upconv8 = convreluss(64, 1, 3, 1, 1)  # size:256

        self.init_weights()

    '''
    Scenario 1 Enter parameters：[build, antenna]
    Scenario 2 Enter parameters：[build, antenna, samples]
    Scenario 3 Enter parameters：[samples, mask]
    '''
    def forward(self, build, antenna):
        # inputs
        x = torch.cat([build, antenna], dim=1)

        # PHASE 1

        # encoder
        layer_conv1 = self.conv1(x)
        layer_conv1 = self.att1(layer_conv1)
        layer_AOT1 = self.middle1(layer_conv1)

        layer_conv2 = self.conv2(layer_AOT1)
        layer_conv2 = self.att2(layer_conv2)
        layer_AOT2 = self.middle2(layer_conv2)

        layer_conv3 = self.conv3(layer_AOT2)
        layer_conv3 = self.att3(layer_conv3)
        layer_AOT3 = self.middle3(layer_conv3)

        layer_conv4 = self.conv4(layer_AOT3)
        layer_conv4 = self.att4(layer_conv4)
        layer_AOT4 = self.middle4(layer_conv4)

        # decoder
        layer_up_conv1 = self.upconv1(layer_AOT4)
        layer_cat1 = self.cat1(torch.cat([layer_up_conv1, layer_conv3], dim=1))

        layer_up_conv2 = self.upconv2(layer_cat1)
        layer_cat2 = self.cat2(torch.cat([layer_up_conv2, layer_conv2], dim=1))

        layer_up_conv3 = self.upconv3(layer_cat2)
        layer_cat3 = self.cat3(torch.cat([layer_up_conv3, layer_conv1], dim=1))

        layer_up_conv4 = self.upconv4(layer_cat3)

        # PHASE 1

        # encoder
        layer_conv5 = self.conv5(layer_up_conv4)
        layer_conv5 = self.att1(layer_conv5)
        layer_AOT5 = self.middle5(layer_conv5)

        layer_conv6 = self.conv6(layer_AOT5)
        layer_conv6 = self.att6(layer_conv6)
        layer_AOT6 = self.middle6(layer_conv6)

        layer_conv7 = self.conv7(layer_AOT6)
        layer_conv7 = self.att7(layer_conv7)
        layer_AOT7 = self.middle7(layer_conv7)

        layer_conv8 = self.conv8(layer_AOT7)
        layer_conv8 = self.att8(layer_conv8)
        layer_AOT8 = self.middle8(layer_conv8)

        # decoder
        layer_up_conv5 = self.upconv5(layer_AOT8)
        layer_cat4 = self.cat4(torch.cat([layer_up_conv5, layer_conv7], dim=1))

        layer_up_conv6 = self.upconv6(layer_cat4)
        layer_cat5 = self.cat5(torch.cat([layer_up_conv6, layer_conv6], dim=1))

        layer_up_conv7 = self.upconv7(layer_cat5)
        layer_cat6 = self.cat6(torch.cat([layer_up_conv7, layer_conv5], dim=1))

        layer_up_conv8 = self.upconv8(layer_cat6)

        # normalization

        x = torch.tanh(layer_up_conv8)

        return x

# Debug
def test():
    x = torch.randn((1, 5, 512, 512))
    model = Generator(BaseNetwork)
    preds = model(x)
    print(preds[0].shape)


if __name__ == "__main__":
    test()


# Define Discriminator

class Discriminator(BaseNetwork):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        # inputs channel
        inc = 1
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat
