from collections import defaultdict

import torch
from torch import optim
from optimizers.misc import validator, Optimizer


class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, initial_accumulator_value=0, eps=1e-10):
        super(Adagrad, self).__init__(params)
        self.lr = lr
        self.eps = eps
        self.state = defaultdict(dict)
        for p in self.params:
            param_state = self.state[p]
            param_state['sum'] = torch.full_like(p, fill_value=initial_accumulator_value)

    def step(self):
        for p in self.params:
            if p.grad is not None:
                param_state = self.state[p]
                g = torch.clone(p.grad).detach()
                s = param_state['sum']
                # I have given two different implementations, the commented one have some error with the official one.
                # s <- s + g^2
                # s.add_(g * g)
                s.addcmul_(g, g, value=1)
                # p <- p - η / sqrt(s) * g
                # p.data -= self.lr / (torch.sqrt(s) + self.eps) * g
                std = s.sqrt().add_(self.eps)
                p.data.addcdiv_(g, std, value=-self.lr)


if __name__ == '__main__':
    param_dict = {
        'lr': 0.001,  # the official value for lr is 0.01
        'initial_accumulator_value': 0,
        'eps': 1e-10
    }
    validator(optim.Adagrad, Adagrad, param_dict)
