import torch
from torch import nn, Tensor, autograd
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def saturating_adv_loss(d_real_out, d_fake_out):
    # d_loss part is as same as the non-saturating one
    d_real_loss = - torch.log(d_real_out).mean()
    d_fake_loss = - torch.log(1 - d_fake_out).mean()
    d_loss = (d_real_loss + d_fake_loss) / 2
    # g_loss = log(1 - d_fake_out)
    g_loss = torch.log(1 - d_fake_out).mean()
    return d_loss, g_loss


# BCE version of the origin adv loss (non-saturating adv loss)
def bce_adv_loss(d_real_out, d_fake_out):
    # d_real_loss = - log(d_real_out)
    d_real_loss = F.binary_cross_entropy(d_real_out, torch.full_like(d_real_out, 1))
    # d_fake_loss = - log(1 - d_real_out)
    d_fake_loss = F.binary_cross_entropy(d_fake_out, torch.full_like(d_fake_out, 0))
    d_loss = (d_real_loss + d_fake_loss) / 2
    # g_loss = - log(d_fake_out)
    g_loss = F.binary_cross_entropy(d_fake_out, torch.full_like(d_fake_out, 1))
    return d_loss, g_loss


# BCE_with_logit version of the origin adv loss (non-saturating adv loss)
def bce_with_logit_adv_loss(d_real_logit, d_fake_logit):
    d_real_loss = F.binary_cross_entropy_with_logits(d_real_logit, torch.full_like(d_real_logit, 1))
    d_fake_loss = F.binary_cross_entropy_with_logits(d_fake_logit, torch.full_like(d_fake_logit, 0))
    d_loss = (d_real_loss + d_fake_loss) / 2
    g_loss = F.binary_cross_entropy_with_logits(d_fake_logit, torch.full_like(d_fake_logit, 1))
    return d_loss, g_loss


# log version of the origin adv loss (non-saturating adv loss), actually it's as same as bce_adv_loss
def log_adv_loss(d_real_out, d_fake_out):
    d_real_loss = - torch.log(d_real_out).mean()
    d_fake_loss = - torch.log(1 - d_fake_out).mean()
    d_loss = (d_real_loss + d_fake_loss) / 2
    g_loss = - torch.log(d_fake_out).mean()
    return d_loss, g_loss


# softplus adv loss (non-saturating adv loss), actually it's as same as bce_adv_loss
def softplus_adv_loss(d_real_logit, d_fake_logit):
    d_real_loss = F.softplus(-d_real_logit).mean()  # - log(d_real_out)
    d_fake_loss = F.softplus(d_fake_logit).mean()  # - log(1 - d_fake_out)
    d_loss = (d_real_loss + d_fake_loss) / 2
    # g_loss = softplus(-d_fake_logit)
    #        = log(1 + e^(-d_fake_logit))
    #        = -log(1 / (1 + e^(-d_fake_logit)))
    #        = -log(sigmoid(d_fake_logit))
    #        = -log(d_fake_out)
    g_loss = F.softplus(-d_fake_logit).mean()
    return d_loss, g_loss


# hinge adv loss
def hinge_adv_loss(d_real_logit, d_fake_logit):
    """
    Self-Attention Generative Adversarial Networks (https://arxiv.org/abs/1805.08318)
    """
    d_real_loss = F.relu(1 - d_real_logit).mean()
    d_fake_loss = F.relu(1 + d_fake_logit).mean()
    d_loss = (d_real_loss + d_fake_loss) / 2
    g_loss = F.relu(1 - d_fake_logit).mean()
    return d_loss, g_loss


# WGAN's adv loss
def wgan_adv_loss(d_real_out, d_fake_out):
    d_loss = - d_real_out.mean() + d_fake_out.mean()
    g_loss = - d_fake_out.mean()
    # Beside this, W-GAN apply clipping to discriminator's weight
    # for p in discriminator.parameters():
    #     p.data.clamp_(-opt.clip_value, opt.clip_value)
    # Reference: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py#L140
    return d_loss, g_loss


# WGAN-GP's adv loss
def wgan_gp_adv_loss(d_real_out, d_fake_out, discriminator, real_samples, fake_samples, lambda_gp=1):
    # add gradient_penalty to the WGAN's adv loss of discriminator
    d_loss = - d_real_out.mean() + d_fake_out.mean() \
             + lambda_gp * calculate_gradient_penalty(discriminator, real_samples, fake_samples)
    # g_loss part is as same as the WGAN's adv loss of generator
    g_loss = - d_fake_out.mean()
    return d_loss, g_loss


def calculate_gradient_penalty(discriminator, real_samples, fake_samples):
    # https://github.com/eriklindernoren/PyTorch-GAN/blob/36d3c77e5ff20ebe0aeefd322326a134a279b93e/implementations/wgan_gp/wgan_gp.py#L119
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def is_same_value(a, b):
    return abs(a - b) < 1e-5


if __name__ == '__main__':
    real_logit = torch.randn(4)
    fake_logit = torch.randn(4)
    print("logit", real_logit, fake_logit)
    real_out = torch.sigmoid(real_logit)
    fake_out = torch.sigmoid(fake_logit)
    print("out", real_out, fake_out)
    d_loss1, g_loss1 = bce_adv_loss(real_out, fake_out)
    d_loss2, g_loss2 = log_adv_loss(real_out, fake_out)
    print("bce_adv_loss", d_loss1, g_loss1)
    print("log_adv_loss", d_loss2, g_loss2)
    assert is_same_value(d_loss1, d_loss2) and is_same_value(g_loss1, g_loss2)
    d_loss3, g_loss3 = saturating_adv_loss(real_out, fake_out)
    print("saturating_adv_loss", d_loss3, g_loss3)
    d_loss4, g_loss4 = wgan_adv_loss(real_out, fake_out)
    print("wgan_adv_loss", d_loss4, g_loss4)
    d_loss5, g_loss5 = softplus_adv_loss(real_logit, fake_logit)
    print("softplus_adv_loss", d_loss5, g_loss5)
    assert is_same_value(d_loss1, d_loss5) and is_same_value(g_loss1, g_loss5)
    d_loss6, g_loss6 = bce_with_logit_adv_loss(real_logit, fake_logit)
    print("bce_with_logit_adv_loss", d_loss6, g_loss6)
    assert is_same_value(d_loss1, d_loss6) and is_same_value(g_loss1, g_loss6)
    d_loss7, g_loss7 = hinge_adv_loss(real_logit, fake_logit)
    print("hinge_adv_loss", d_loss7, g_loss7)
