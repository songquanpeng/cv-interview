from torch import optim
from optimizers.misc import validator, Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        super(SGD, self).__init__(params)

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad.data


if __name__ == '__main__':
    param_dict = {
        'lr': 0.001
    }
    validator(SGD, optim.SGD, param_dict)
