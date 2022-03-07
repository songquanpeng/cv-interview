def SGD(params, lr):
    for p in params:
        p.data -= lr * p.grad.data
