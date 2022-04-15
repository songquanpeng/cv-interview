# https://blog.csdn.net/Biyoner/article/details/88916247
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def old_conv2d(input, weight, stride, padding):
    """
    this implementation does not support batched input & multi kernels
    and it's input is narray
    """
    C, H, W = input.shape
    KC, K, _ = weight.shape
    assert C == KC, "unmatched kernel channel number & input channel number"
    result_height = int((H - K + 2 * padding) / stride) + 1
    result_width = int((W - K + 2 * padding) / stride) + 1
    padded_input = np.zeros((C, H + 2 * padding, W + 2 * padding))
    for i in range(C):
        padded_input[i, padding:padding + H, padding:padding + W] = input[i, :, :]
    flatted_input = np.zeros((result_height * result_width, KC * K * K))
    flatted_weight = weight.reshape(-1, 1)
    row = 0
    for i in range(result_height):
        for j in range(result_width):
            roi = padded_input[:, i * stride:i * stride + K, j * stride:j * stride + K]
            flatted_input[row] = roi.reshape(-1)
            row += 1
    result = np.dot(flatted_input, flatted_weight).reshape(result_height, result_width)
    return result


def conv2d(input, weight, stride, padding, bias=None):
    """
    input: [N, C, H, W]
    weight: [OUT_C, IN_C, KW, KH]
    bias: [OUT_C]
    """
    N, C, H, W = input.shape
    OUT_C, IN_C, KW, KH = weight.shape
    assert C == IN_C, "unmatched input & weight"
    padded_input = torch.zeros((N, C, H + 2 * padding, W + 2 * padding))
    for n in range(N):
        for c in range(C):
            padded_input[n, c, padding:padding + H, padding:padding + W] = input[n, c, :, :]
    OUT_H = int((H - KH + 2 * padding) / stride) + 1
    OUT_W = int((W - KW + 2 * padding) / stride) + 1
    output = torch.zeros((N, OUT_C, OUT_H, OUT_W))
    flatted_input = torch.zeros((N, OUT_C, OUT_H * OUT_W, IN_C * KH * KW))
    flatted_weight = weight.view((OUT_C, IN_C * KH * KW))
    for n in range(N):
        for out_c in range(OUT_C):
            row = 0
            for out_h in range(OUT_H):
                for out_w in range(OUT_W):
                    roi = padded_input[n, :, out_h * stride:out_h * stride + KH, out_w * stride:out_w * stride + KW]
                    flatted_input[n, out_c, row, :] = roi.reshape(-1)
                    row += 1
            result = flatted_input[n, out_c] @ flatted_weight[out_c]
            output[n, out_c, :, :] = result.reshape(OUT_H, OUT_W)
    if bias:
        bias = bias.reshape(1, OUT_C, 1, 1)
        output += bias
    return output


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return conv2d(x, self.weight, stride=self.stride, padding=self.padding, bias=self.bias)


if __name__ == '__main__':
    input_data = [
        [
            [1, 0, 1, 2, 1],
            [0, 2, 1, 0, 1],
            [1, 1, 0, 2, 0],
            [2, 2, 1, 1, 0],
            [2, 0, 1, 2, 0],
        ],
        [
            [2, 0, 2, 1, 1],
            [0, 1, 0, 0, 2],
            [1, 0, 0, 2, 1],
            [1, 1, 2, 1, 0],
            [1, 0, 1, 1, 1],
        ]
    ]
    weight_data = [
        [
            [1, 0, 1],
            [-1, 1, 0],
            [0, -1, 0],
        ],
        [
            [-1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    ]
    input = torch.tensor(np.asarray(input_data, np.float32)).unsqueeze(0)
    weight = torch.tensor(np.asarray(weight_data, np.float32)).unsqueeze(0)
    my_res = conv2d(input, weight, stride=1, padding=0)
    official_res = F.conv2d(input, weight, stride=1, padding=0)
    print(my_res)
    print(official_res)
    assert torch.equal(my_res, official_res)
    my_conv = Conv2d(2, 1, 3, 1, 0)
    official_conv = nn.Conv2d(2, 1, 3, 1, 0)
    my_res = my_conv(input)
    official_res = official_conv(input)
    print(my_res)
    print(official_res)
    # assert torch.equal(my_res, official_res)
