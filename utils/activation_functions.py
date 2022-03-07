import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))


def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = exp_x.sum(keepdims=True)
    return exp_x / sum_exp_x


def relu(x):
    return np.maximum(x, 0)


def leaky_relu(x, negative_slope=0.01):
    return np.where(x > 0, x, x * negative_slope)


def softplus(x):
    return np.log(1 + np.exp(x))


if __name__ == '__main__':
    def main():
        np.set_printoptions(suppress=True)
        a = np.array([1, 4, 5, 9])
        print(softmax(a))

        x = np.linspace(-5, 5, 100)
        plt.plot(x, sigmoid(x), label='sigmoid')
        plt.plot(x, tanh(x), label='tanh')
        plt.plot(x, relu(x), label='relu')
        plt.plot(x, leaky_relu(x), label='leaky_relu')
        plt.plot(x, softplus(x), label='softplus')
        plt.legend()
        plt.show()

    main()
