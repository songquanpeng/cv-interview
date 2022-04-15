import torch
from torch import autograd


# https://github.com/NVlabs/stylegan2-ada-pytorch/blob/6f160b3d22b8b178ebe533a50d4d5e63aedba21d/training/loss.py#L123
def r1_regularization(real_images, d_real_logit):
    # https://pytorch.org/docs/stable/generated/torch.autograd.grad.html
    r1_grads = torch.autograd.grad(outputs=[d_real_logit.sum()], inputs=[real_images],
                                   create_graph=True, only_inputs=True)[0]
    r1_penalty = 0.5 * r1_grads.square().sum([1, 2, 3])
    return r1_penalty


# https://github.com/saic-mdal/CIPS/blob/HEAD/train.py
def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


# https://github.com/clovaai/stargan-v2/blob/HEAD/core/solver.py
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
