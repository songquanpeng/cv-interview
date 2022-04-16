import numpy as np
import argparse
from copy import deepcopy
from utils.data import get_MNIST_loader
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, keepdims=True)
    return ex / sum_ex


def d_softmax(x):
    # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
    softmax_x = softmax(x)
    diag_softmax_x = np.diag(softmax_x)
    matrix = np.outer(softmax_x, softmax_x.T)  # the outer product
    jacobi_matrix = - matrix + diag_softmax_x
    return jacobi_matrix


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def cross_entropy(y, y_hat):
    res = - y * np.log(y_hat)
    res = res.sum()
    return res


def d_cross_entropy(y, y_hat):
    return - y / y_hat


def d_softmax_cross_entropy(y, y_hat):
    # https://blog.csdn.net/jasonleesjtu/article/details/89426465
    return y_hat - y


def d_softmax_cross_entropy2(y, y_hat, z):
    """
    z is the last layer's unactivated output
    """
    dce_ds = d_cross_entropy(y, y_hat)
    ds_dz = d_softmax(z)
    res = []
    for i in range(len(z)):
        tmp = dce_ds * ds_dz.T[i]
        res.append(tmp.sum())
    res = np.array(res)
    return res


class MLP:
    def __init__(self, in_dim, middle_dim, out_dim):
        self.weights = [
            np.random.randn(in_dim, middle_dim),
            np.random.randn(middle_dim, out_dim)
        ]
        self.biases = [
            np.zeros(middle_dim),
            np.zeros(out_dim)
        ]
        self.weights_grad = deepcopy(self.weights)
        self.biases_grad = deepcopy(self.biases)
        self.zero_grad()
        self.unactivated_outputs = []
        self.activated_outputs = []

    def zero_grad(self):
        for grad in self.weights_grad:
            grad.fill(0)
        for grad in self.biases_grad:
            grad.fill(0)

    def forward(self, x):
        feature = x
        self.unactivated_outputs = []
        self.activated_outputs = [x]
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            output = feature @ weight + bias
            self.unactivated_outputs.append(output)
            if i != len(self.weights) - 1:  # middle layer's activation
                feature = sigmoid(output)
            else:  # last layer's activation
                feature = softmax(output)
            self.activated_outputs.append(feature)
        return feature

    def __call__(self, x):
        return self.forward(x)

    def backward(self, y, y_hat):
        # z_last is the last layer's unactivated output
        z_last = self.unactivated_outputs[-1]
        # delta_last is the gradient of the last layer's unactivated output
        delta_last = d_softmax_cross_entropy(y, y_hat)
        # delta_last2 = d_softmax_cross_entropy2(y, y_hat, z_last)
        # assert delta_last == delta_last2, "the two ways should give the same result"
        weight_last = self.weights[-1]
        # z_middle the middle layer's unactivated output
        z_middle = self.unactivated_outputs[-2]
        delta_middle = weight_last @ delta_last * d_sigmoid(z_middle)

        self.weights_grad[-1] += np.outer(self.activated_outputs[-2], delta_last)
        self.biases_grad[-1] += delta_last
        self.weights_grad[-2] += np.outer(self.activated_outputs[-3], delta_middle)
        self.biases_grad[-2] += delta_middle

    def update(self, learning_rate=0.0001):
        for weight, grad in zip(self.weights, self.weights_grad):
            weight -= learning_rate * grad
        for bias, grad in zip(self.biases, self.biases_grad):
            bias -= learning_rate * grad


def train(args):
    mlp = MLP(args.in_dim, args.middle_dim, args.out_dim)
    dataloader = get_MNIST_loader()
    all_losses = []
    for epoch in range(args.epoch_num):
        losses = []
        for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = x.flatten().numpy()
            tmp = np.zeros(10)
            tmp[y] = 1
            y = tmp
            y_hat = mlp(x)
            loss = cross_entropy(y, y_hat)
            losses.append(loss)
            # print(f"epoch: {epoch:06} iter: {i:06} loss: {loss:.6f}")
            mlp.zero_grad()
            mlp.backward(y, y_hat)
            mlp.update()
        losses = np.array(losses).mean()
        print(f"Epoch mean loss: {losses:.6f}")
        all_losses.append(losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--in_dim', type=int, default=28 * 28)
    parser.add_argument('--middle_dim', type=int, default=100)
    parser.add_argument('--out_dim', type=int, default=10)
    opt = parser.parse_args()
    train(opt)
