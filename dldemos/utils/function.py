import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def relu_de(x):
    return np.where(x > 0, 1, 0)
