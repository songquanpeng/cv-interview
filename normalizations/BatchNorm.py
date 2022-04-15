# https://github.com/GYee/CV_interviews_Q-A/blob/master/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89/04_BN%E5%B1%82%E7%9A%84%E6%B7%B1%E5%85%A5%E7%90%86%E8%A7%A3.md#%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0bn%E5%B1%82

import torch
from torch import nn


def batch_norm(feature_map, gamma, beta, is_training, running_mean, running_var, momentum=0.1, epsilon=1e-5):
    if is_training:
        if len(feature_map.shape) == 4:
            # feature_map: [N, C, H, W]
            cur_mean = feature_map.mean(dim=[0, 2, 3], keepdim=True)
            cur_var = feature_map.var(dim=[0, 2, 3], keepdim=True)
        else:
            # feature_map: [N, F]
            cur_mean = feature_map.mean(dim=[0], keepdim=True)
            cur_var = feature_map.var(dim=[0], keepdim=True)
        running_mean = (1 - momentum) * running_mean + momentum * cur_mean
        running_var = (1 - momentum) * running_var + momentum * cur_var
    else:
        cur_mean = running_mean
        cur_var = running_var
    # normalize the feature map
    feature_map = (feature_map - cur_mean) / torch.sqrt(cur_var + epsilon)
    # scale and shift
    feature_map = feature_map * gamma + beta
    return feature_map, running_mean, running_var


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims, momentum=0.1, epsilon=1e-5):
        self.momentum = momentum
        self.epsilon = epsilon
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        running_mean = torch.zeros(shape)
        running_var = torch.ones(shape)  # in the official implementation, they are ones
        self.register_buffer('running_mean', running_mean)
        self.register_buffer('running_var', running_var)

    def forward(self, x):
        out, running_mean, running_var = batch_norm(x, self.gamma, self.beta, self.training, self.running_mean,
                                                    self.running_var, self.momentum, self.epsilon)
        if self.training:
            self.running_mean = running_mean
            self.running_var = running_var
        return out


if __name__ == '__main__':
    batch_size, num_features, width, height = 4, 12, 16, 16
    dummy_feature = torch.randn((batch_size, num_features, width, height))
    my_bn = BatchNorm(num_features, 4)
    official_bn = nn.BatchNorm2d(num_features)
    my_res = my_bn(dummy_feature)
    official_res = official_bn(dummy_feature)
    assert torch.allclose(my_bn.running_mean.squeeze(), official_bn.running_mean)
    assert torch.allclose(my_bn.running_var.squeeze(), official_bn.running_var)
    print(my_res.mean(), my_res.var())
    print(official_res.mean(), official_res.var())
