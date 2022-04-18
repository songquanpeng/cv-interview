from collections import defaultdict

import torch
from torch import optim
from optimizers.misc import validator, Optimizer


class Momentum(Optimizer):
    def __init__(self, params, lr, momentum):
        super(Momentum, self).__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.state = defaultdict(dict)

    def step(self):
        for p in self.params:
            param_state = self.state[p]
            if 'momentum_buffer' not in param_state:
                param_state['momentum_buffer'] = torch.clone(p.grad).detach()
                buffer = param_state['momentum_buffer']
            else:
                buffer = param_state['momentum_buffer']
                # v <- γ * v + g
                buffer.mul_(self.momentum).add_(p.grad)
            # g <- g - η * v
            p.data -= self.lr * buffer


if __name__ == '__main__':
    param_dict = {
        'lr': 0.001,
        'momentum': 0.9
    }
    validator(optim.SGD, Momentum, param_dict)
