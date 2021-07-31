import torch
from torch.nn import functional as F


# https://github.com/saic-mdal/CIPS/blob/HEAD/train.py
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


# https://github.com/saic-mdal/CIPS/blob/HEAD/train.py
def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


# https://github.com/clovaai/stargan-v2/blob/HEAD/core/solver.py
# loss_fake = adv_loss(out, 0)
# loss_real = adv_loss(out, 1)
def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
def loss_dcgan_dis(dis_fake, dis_real):
    L1 = torch.mean(F.softplus(-dis_real))
    L2 = torch.mean(F.softplus(dis_fake))
    return L1, L2


# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
def loss_dcgan_gen(dis_fake):
    loss = torch.mean(F.softplus(-dis_fake))
    return loss


# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


# https://github.com/ajbrock/BigGAN-PyTorch/blob/master/losses.py
def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss
