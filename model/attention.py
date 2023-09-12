
# 注意力机制
import torch
import torch.nn as nn
import math

# 通道注意力机制的典型代表:se_Net
class se_Net(nn.Module):
    def __init__(self, channel, ratio=16):
        super(se_Net, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# eca 注意力机制模块
class eca_Net(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_Net, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

# 空间注意力机制
class Spatial_Attention(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute attention weights
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        weights = self.sigmoid(x2)

        # Apply attention weights
        out = x * weights

        return out

# CBAM注意力机制(下面全是)
class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x




class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        padding = kernel_size //2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out * x


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):  # 注：通道数大于等于ratio才可用
        super(cbam_block, self).__init__()
        self.channelattention = channel_attention(channel, ratio=ratio)
        self.spatialattention = spatial_attention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channelattention(x)
        out = self.spatialattention(out)
        return out

class cbam_block1(nn.Module):
    def __init__(self, channel, ratio=1, kernel_size=7):  # 注：通道数大于等于ratio才可用
        super(cbam_block1, self).__init__()
        self.channelattention = channel_attention(channel, ratio=ratio)
        self.spatialattention = spatial_attention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channelattention(x)
        out = self.spatialattention(out)
        return out

# # TEST:数字代表通道数
# model = cbam_block1(1)
# inputs = torch.ones([15,1,256,256])
# outputs = model(inputs)
# print(outputs.shape)