import torch
from torch.nn import functional as F


def log_softmax(x, dim=1):
    # softmax(i) = e^xi / sum(e^x)
    # log softmax(i) = xi - log(sum(e^x))
    exp_x = torch.exp(x)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return x - torch.log(sum_exp_x)


def nll_loss(a, b):
    # negative log likelihood loss
    if b.dim() == 1:
        # turn indices to one hot like
        mask = F.one_hot(b, num_classes=a.shape[1])
    else:
        # already per-class probability (shape [N, C])
        mask = b
    loss = (- a * mask).sum(dim=1).mean()
    return loss


def cross_entropy(logits, targets):
    # please notice, PyTorch's F.cross_entropy always take logits as input
    return nll_loss(log_softmax(logits, 1), targets)


def cross_entropy2(logits, one_hot_targets):
    softmax_logit = F.softmax(logits, dim=1)
    loss = - (one_hot_targets * torch.log(softmax_logit)).sum(dim=1)
    return loss.mean()


if __name__ == '__main__':
    torch.manual_seed(0)
    logit = torch.randn(2, 4)
    target = torch.tensor([1, 3])
    one_hot_target = F.one_hot(target, num_classes=logit.shape[1]).float()
    softmax_logit = F.softmax(logit, dim=1)
    log_softmax_logit = F.log_softmax(logit, dim=1)
    r1 = nll_loss(log_softmax_logit, target)
    r2 = F.nll_loss(log_softmax_logit, target)
    print(r1, r2)
    assert abs(r1 - r2) < 1e-5
    r1 = log_softmax(logit, dim=1)
    r2 = F.log_softmax(logit, dim=1)
    print(r1, "\n", r2)
    assert abs(r1.var() - r2.var()) < 1e-5
    r1 = cross_entropy(logit, one_hot_target)
    r2 = cross_entropy(logit, target)
    r3 = cross_entropy2(logit, one_hot_target)
    r4 = F.cross_entropy(logit, target)
    print(r1, r2, r3, r4)
    assert abs(r1 - r2) < 1e-5 and abs(r2 - r3) < 1e-5 and abs(r3 - r4) < 1e-5
