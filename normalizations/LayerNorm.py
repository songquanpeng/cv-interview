# https://www.zhihu.com/question/454292446/answer/1837760076
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """亦可见nn.LayerNorm"""

    def __init__(self, features, epsilon=1e-6):
        """features: normalized_shape
            epsilon: 一个很小的数，防止数值计算的除0错误
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x):
        """
        Args:
            x: 输入序列张量，形状为[B, L, D]
        """
        mean = x.mean(-1, keepdim=True)  # 在X的最后一个维度求均值
        std = x.std(-1, keepdim=True)  # 在X的最后一个维度求方差
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta
