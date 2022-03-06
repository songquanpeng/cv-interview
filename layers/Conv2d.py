# https://blog.csdn.net/Biyoner/article/details/88916247
import numpy as np
import torch
import torch.nn.functional as F


def conv2d(input, weight, stride, padding):
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

        ],
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
    input = np.asarray(input_data, np.float32)
    weight = np.asarray(weight_data, np.float32)
    print(conv2d(input, weight, stride=1, padding=0))
    print(F.conv2d(torch.tensor(input).unsqueeze(0), torch.tensor(weight).unsqueeze(0), stride=1,
                   padding=0).squeeze().numpy())
