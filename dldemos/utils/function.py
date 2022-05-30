import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_de(x):
    tmp = sigmoid(x)
    return tmp * (1 - tmp)


def relu(x):
    return np.maximum(x, 0)


def relu_de(x):
    return np.where(x > 0, 1, 0)


def get_activation_func(name):
    if name == 'sigmoid':
        return sigmoid
    elif name == 'relu':
        return relu
    else:
        raise KeyError(f'No such activavtion function {name}')


def get_activation_de_func(name):
    if name == 'sigmoid':
        return sigmoid_de
    elif name == 'relu':
        return relu_de
    else:
        raise KeyError(f'No such activavtion function {name}')
