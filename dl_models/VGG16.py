import torch
from torch import nn
from utils.model import count_parameters
import torchvision.models as models


class VGGConvBlock(nn.Module):
    def __init__(self, num_convs, in_dim, out_dim, kernel_size=3, stride=1, padding=1):
        # please notice the padding is 1, cause the conv layers here don't reduce the size of the feature map
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.extend([
                nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding),
                nn.ReLU(inplace=True),
            ])
            in_dim = out_dim
        layers.append(nn.MaxPool2d(2, 2))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=1000, in_dim=3, init_dim=64, max_dim=512, dropout=0.5):
        super().__init__()
        layers = []
        out_dim = init_dim
        for num_convs in [2, 2, 3, 3, 3]:
            layers.append(VGGConvBlock(num_convs, in_dim, out_dim))
            in_dim = out_dim
            out_dim = min(out_dim * 2, max_dim)
        self.conv = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(max_dim * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    vgg = VGG16()
    count_parameters(vgg, 'VGG16')
    dummy_x = torch.randn((1, 3, 224, 224))
    output = vgg(dummy_x)
    official_vgg = models.vgg16()
    count_parameters(vgg, 'Official VGG16')
