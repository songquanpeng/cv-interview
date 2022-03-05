import torch
from torch import nn


class InstanceNorm(nn.Module):
    def __init__(self, feature_shape, epsilon=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(feature_shape))
        self.beta = nn.Parameter(torch.zeros(feature_shape))
        self.epsilon = epsilon

    def forward(self, x):
        """
        x: [N, C, H, W]
        """
        N, C, H, W = x.shape
        mean = x.view(N, C, -1).mean(-1, keepdim=True)
        std = x.view(N, C, -1).std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
