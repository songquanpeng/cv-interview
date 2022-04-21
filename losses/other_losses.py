import torch
import torch.nn.functional as F


def l1_loss(a, b):
    # L1 / mean absolute error / MAE
    # L1 = mean(abs(a-b)).mean()
    # I don't know why not L1 = sum(abs(ai-bi)).mean()
    return torch.abs(a - b).mean(dim=1).mean()


def l2_loss(a, b):
    # L2 = sqrt(sum[(ai-bi)^2])
    # Actually it's not the same thing with mse
    return torch.square(a - b).sum(dim=1).sqrt().mean()


def mse_loss(a, b):
    # mean squared error / MSE / squared Euclidean norm / squared L2 norm
    # The squared L2 norm is convenient
    # because it removes the square root and we end up with the simple sum of every squared value of the vector.
    return torch.square(a - b).mean(dim=1).mean()


def triplet_margin_with_distance_loss(anchor, positive, negative, distance_function, margin=1.0):
    loss = distance_function(anchor, positive) - distance_function(anchor, negative) + margin
    loss *= loss > 0
    return loss.mean()


def triplet_margin_loss(anchor, positive, negative, margin=1.0):
    return triplet_margin_with_distance_loss(anchor, positive, negative, l2_loss, margin)


def smooth_l1_loss(a, b, beta=1.0):
    # https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss
    mask = abs(a - b) < beta
    part1 = 0.5 * torch.square(a - b) / beta
    part2 = abs(a - b) - 0.5 * beta
    loss = part1 * mask + part2 * ~mask
    return loss.mean()


if __name__ == '__main__':
    torch.manual_seed(0)
    a = torch.randn((2, 4))
    b = torch.randn((2, 4))
    c = torch.randn((2, 4))
    r1 = triplet_margin_loss(a, b, c)
    r2 = F.triplet_margin_loss(a, b, c)
    print(r1, r2)
    assert abs(r1 - r2) < 1e-5
    r1 = l1_loss(a, b)
    r2 = F.l1_loss(a, b)
    print(r1, r2)
    assert abs(r1 - r2) < 1e-5
    r1 = mse_loss(a, b)
    r2 = F.mse_loss(a, b)
    print(r1, r2)
    assert abs(r1 - r2) < 1e-5
    r1 = smooth_l1_loss(a, b)
    r2 = F.smooth_l1_loss(a, b)
    print(r1, r2)
    assert abs(r1 - r2) < 1e-5
