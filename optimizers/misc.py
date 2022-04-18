import torch
from copy import deepcopy
from dl_models.ResNet import ResNet18
import torch.nn.functional as F


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        # actually there are more things to do in the official zero_grad()
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        raise NotImplementedError


def validator(Optim1, Optim2, param_dict):
    dummy_x = torch.randn((2, 3, 224, 224))
    y = F.one_hot(torch.arange(0, 2), num_classes=10).float()
    net1 = ResNet18(10, 3)
    net2 = deepcopy(net1)
    my_optim = Optim1(net1.parameters(), **param_dict)
    official_optim = Optim2(net2.parameters(), **param_dict)
    for i in range(10):
        out1 = net1(dummy_x)
        out2 = net2(dummy_x)
        loss1 = F.binary_cross_entropy_with_logits(out1, y)
        loss2 = F.binary_cross_entropy_with_logits(out2, y)
        print(loss1.item(), loss2.item())
        assert abs(loss1.item() - loss2.item()) < 1e-5
        my_optim.zero_grad()
        official_optim.zero_grad()
        loss1.backward()
        loss2.backward()
        my_optim.step()
        official_optim.step()
