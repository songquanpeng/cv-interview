from collections import defaultdict

import torch
from torch import optim
from optimizers.misc import validator, Optimizer


class Adadelta(Optimizer):
    def __init__(self, params, lr=1, rho=0.9, eps=1e-6):
        super(Adadelta, self).__init__(params)
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.state = defaultdict(dict)
        for p in self.params:
            param_state = self.state[p]
            param_state['square_avg'] = torch.zeros_like(p)
            param_state['acc_delta'] = torch.zeros_like(p)  # accumulative delta

    def step(self):
        for p in self.params:
            if p.grad is not None:
                param_state = self.state[p]
                g = torch.clone(p.grad).detach()
                square_avg = param_state['square_avg']
                acc_delta = param_state['acc_delta']
                # s <- ρ * s + (1 - ρ) * g^2
                square_avg.mul_(self.rho).addcmul_(g, g, value=1 - self.rho)
                # g' <- sqrt(Δp/s) * g
                std = square_avg.add(self.eps).sqrt_()
                delta = acc_delta.add(self.eps).sqrt_().div_(std).mul_(g)
                # p <- p - g'
                p.data.add_(delta, alpha=-self.lr)
                # Δp <- ρ * Δp + (1 - ρ) * g'^2
                acc_delta.mul_(self.rho).addcmul_(delta, delta, value=1 - self.rho)


if __name__ == '__main__':
    param_dict = {
        'lr': 1,  # shouldn keep the learning rate as 1
        'rho': 0.9,
        'eps': 1e-6
    }
    validator(optim.Adadelta, Adadelta, param_dict)
