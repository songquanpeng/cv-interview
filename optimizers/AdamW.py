import math
from collections import defaultdict

import torch
from torch import optim
from optimizers.misc import validator, Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8):
        super(AdamW, self).__init__(params)
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.state = defaultdict(dict)
        for p in self.params:
            param_state = self.state[p]
            param_state['avg_grad'] = torch.zeros_like(p)
            param_state['avg_grad_square'] = torch.zeros_like(p)
            param_state['step'] = 0  # parameters may not be updated synchronously, we cannot use a global step

    def step(self):
        for p in self.params:
            if p.grad is not None:
                param_state = self.state[p]
                param_state['step'] += 1
                step = param_state['step']
                bias_correction1 = 1 - self.beta1 ** step
                bias_correction2 = 1 - self.beta2 ** step
                grad = torch.clone(p.grad).detach()
                if self.weight_decay != 0:
                    # Adam's weight decay:
                    # grad.add_(p, alpha=self.weight_decay)
                    # AdamW's weight decay:
                    p.data.mul_(1 - self.lr * self.weight_decay)
                v = param_state['avg_grad']
                s = param_state['avg_grad_square']
                # v <- β1 * v + (1 - β1) * grad
                v.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                # s <- β2 * s + (1 - β2) * (grad * grad)
                s.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                # p <- p - η * v / sqrt(s)
                # p <- p - η * (v / bias_correction1) / sqrt(s / bias_correction2)
                p.data.addcdiv_(v, (s.sqrt() / math.sqrt(bias_correction2)).add_(self.eps),
                                value=-self.lr / bias_correction1)


if __name__ == '__main__':
    param_dict = {
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-2
    }
    validator(optim.AdamW, AdamW, param_dict)
