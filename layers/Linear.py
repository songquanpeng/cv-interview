# https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
import torch
from torch import nn
import torch.nn.functional as F


def linear(input, weight, bias=None):
    """
    input: [N, IN_DIM]
    weight: [OUT_DIM, IN_DIM]
    bias: [OUT_DIM]
    """
    output = input @ weight.T
    if bias is not None:
        bias = bias.view(1, -1)
        output += bias
    return output


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return linear(x, self.weight, self.bias)


if __name__ == '__main__':
    input = torch.randn((2, 18))
    weight = torch.randn((4, 18))
    bias = torch.randn(4)
    my_res = linear(input, weight, bias)
    official_res = F.linear(input, weight, bias)
    print(my_res)
    print(official_res)
    assert torch.equal(my_res, official_res)
    my_linear = Linear(18, 4)
    official_linear = nn.Linear(18, 4)
    my_linear.weight = official_linear.weight
    my_linear.bias = official_linear.bias
    my_res = my_linear(input)
    official_res = official_linear(input)
    print(my_res)
    print(official_res)
    assert torch.equal(my_res, official_res)
