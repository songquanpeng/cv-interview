from collections import defaultdict

import torch
from torch import optim
from optimizers.misc import validator, Optimizer


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, momentum=0):
        super(RMSprop, self).__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.state = defaultdict(dict)
        for p in self.params:
            param_state = self.state[p]
            param_state['sum'] = torch.zeros_like(p)

    def step(self):
        for p in self.params:
            if p.grad is not None:
                param_state = self.state[p]
                g = torch.clone(p.grad).detach()
                s = param_state['sum']
                # s <- γ * s + (1 - γ) * g^2
                s.mul_(self.alpha).addcmul_(g, g, value=1 - self.alpha)
                # p <- p - η / sqrt(s) * g
                # p.data -= self.lr / (torch.sqrt(s) + self.eps) * g
                std = s.sqrt().add_(self.eps)
                p.data.addcdiv_(g, std, value=-self.lr)


if __name__ == '__main__':
    param_dict = {
        'lr': 0.001,  # the official value for lr is 0.01
        'alpha': 0.99,
        'eps': 1e-8
    }
    validator(optim.RMSprop, RMSprop, param_dict)
