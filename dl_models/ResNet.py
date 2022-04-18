# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
import torch
from typing import Type, Union, List
from torch import nn
import torch.nn.functional as F
from utils.model import count_parameters
import torchvision.models as models

"""
Notes:
1. Conv -> Norm -> Activation (think about after combining the two normed output, we apply activation to the sum).
2. BasicResBlock: Conv(in, out, 3x3) -> Conv(out, out, 3x3). 
3. BottleneckResBlock: Conv(in, mid, 1x1) -> Conv(mid, mid, 3x3) -> Conv(mid, out, 1x1).
"""


class BasicResBlock(nn.Module):
    """
    The block that used in ResNet-18/34.
    """
    expansion = 1

    def __init__(self, in_channels, planes, stride=1):
        """
        planes: out_channels = planes * 1
        stride: when stride=2, this block is used for downsampling
        """
        super().__init__()
        out_channels = planes * self.expansion
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        self.main = nn.Sequential(*layers)
        self.downsample = None
        # There are two cases totally that we need a non-identical shortcut branch:
        # 1. the main branch has downscaled the feature map's size.
        # 2. the main branch has changed the feature map's channel number.
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.main(x)
        shortcut = x
        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
        # Don't forget to apply activation after we combine the main branch's output and the shortcut branch's output.
        out = F.relu(out)
        return out


class BottleneckResBlock(nn.Module):
    """
    The block that used in ResNet-50/101/152.
    """
    expansion = 4

    def __init__(self, in_channels, planes, stride=1):
        """
        out_channels = planes * 4
        """
        super().__init__()
        out_channels = planes * self.expansion
        layers = [
            nn.Conv2d(in_channels, planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        self.main = nn.Sequential(*layers)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.main(x)
        shortcut = x
        if self.downsample:
            shortcut = self.downsample(x)
        out += shortcut
        # Don't forget to apply activation after we combine the main branch's output and the shortcut branch's output.
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicResBlock, BottleneckResBlock]],
                 num_layers: List[int], num_classes: int, in_channels: int = 3):
        super().__init__()
        # input shape: 3x224x224
        self.in_planes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),  # 64x112x112
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 64x56x56
        )
        self.conv2 = self._make_layer(block, block_num=num_layers[0], planes=64, downsample=False)  # 64x56x56
        self.conv3 = self._make_layer(block, block_num=num_layers[1], planes=128, downsample=True)  # 128x28x28
        self.conv4 = self._make_layer(block, block_num=num_layers[2], planes=256, downsample=True)  # 256x14x14
        self.conv5 = self._make_layer(block, block_num=num_layers[3], planes=512, downsample=True)  # 512x7x7
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, block_num, planes, downsample=True):
        """
        For each block: out_channels = planes * block.expansion
        """
        # if we want downsampling, only the first block will do that
        layers = [block(self.in_planes, planes, stride=2 if downsample else 1)]
        self.in_planes = planes * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        out = self.fc(x.view(x.shape[0], -1))
        return out


def ResNet18(num_classes, img_dim):
    return ResNet(BasicResBlock, [2, 2, 2, 2], num_classes, img_dim)


def ResNet34(num_classes, img_dim):
    return ResNet(BasicResBlock, [3, 4, 6, 3], num_classes, img_dim)


def ResNet50(num_classes, img_dim):
    return ResNet(BottleneckResBlock, [3, 4, 6, 3], num_classes, img_dim)


def ResNet101(num_classes, img_dim):
    return ResNet(BottleneckResBlock, [3, 4, 23, 3], num_classes, img_dim)


def ResNet152(num_classes, img_dim):
    return ResNet(BottleneckResBlock, [3, 8, 36, 3], num_classes, img_dim)


"""
Parameter number of my_resnet18: 11.6895M (11.6895M trainable)
Parameter number of my_resnet34: 21.7977M (21.7977M trainable)
Parameter number of my_resnet50: 25.5570M (25.5570M trainable)
Parameter number of my_resnet101: 44.5492M (44.5492M trainable)
Parameter number of my_resnet152: 60.1928M (60.1928M trainable)
Parameter number of official_resnet18: 11.6895M (11.6895M trainable)
Parameter number of official_resnet34: 21.7977M (21.7977M trainable)
Parameter number of official_resnet50: 25.5570M (25.5570M trainable)
Parameter number of official_resnet101: 44.5492M (44.5492M trainable)
Parameter number of official_resnet152: 60.1928M (60.1928M trainable)
"""

if __name__ == '__main__':
    resnet18 = ResNet18(1000, 3)
    resnet34 = ResNet34(1000, 3)
    resnet50 = ResNet50(1000, 3)
    resnet101 = ResNet101(1000, 3)
    resnet152 = ResNet152(1000, 3)
    dummy_img = torch.randn((1, 3, 224, 224))
    names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    resnets = [resnet18, resnet34, resnet50, resnet101, resnet152]
    prefix = 'my_'
    for resnet, name in zip(resnets, names):
        count_parameters(resnet, prefix + name)
        output = resnet(dummy_img)
    official_resnets = [models.resnet18(), models.resnet34(), models.resnet50(), models.resnet101(), models.resnet152()]
    prefix = 'official_'
    for resnet, name in zip(resnets, names):
        count_parameters(resnet, prefix + name)
        output = resnet(dummy_img)

