import abc
from typing import List

import numpy as np

from dldemos.utils import get_activation_de_func, get_activation_func


class BaseRegressionModel(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, X: np.ndarray, train_mode=True) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, Y: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def gradient_descent(self, learning_rate: float) -> np.ndarray:
        pass

    @abc.abstractmethod
    def save(self, filename: str):
        pass

    @abc.abstractmethod
    def load(self, filename: str):
        pass

    def loss(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        return np.mean(-(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)))

    def evaluate(self, X: np.ndarray, Y: np.ndarray, return_loss=False):
        Y_hat = self.forward(X, train_mode=False)
        Y_hat_predict = np.where(Y_hat > 0.5, 1, 0)
        accuracy = np.mean(np.where(Y_hat_predict == Y, 1, 0))
        if return_loss:
            loss = self.loss(Y, Y_hat)
            return accuracy, loss
        else:
            return accuracy


class DeepNetwork(BaseRegressionModel):

    def __init__(self, neuron_cnt: List[int], activation_func: List[str]):
        assert len(neuron_cnt) - 1 == len(activation_func)
        self.num_layer = len(neuron_cnt) - 1
        self.neuron_cnt = neuron_cnt
        self.activation_func = activation_func
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(self.num_layer):
            self.W.append(
                np.random.randn(neuron_cnt[i + 1], neuron_cnt[i]) * 0.2)
            self.b.append(np.zeros((neuron_cnt[i + 1], 1)))

        self.Z_cache = [None] * self.num_layer
        self.A_cache = [None] * (self.num_layer + 1)
        self.dW_cache = [None] * self.num_layer
        self.db_cache = [None] * self.num_layer

    def forward(self, X, train_mode=True):
        if train_mode:
            self.m = X.shape[1]
        A = X
        self.A_cache[0] = A
        for i in range(self.num_layer):
            Z = np.dot(self.W[i], A) + self.b[i]
            A = get_activation_func(self.activation_func[i])(Z)
            if train_mode:
                self.Z_cache[i] = Z
                self.A_cache[i + 1] = A
        return A

    def backward(self, Y):
        dA = -Y / self.A_cache[-1] + (1 - Y) / (1 - self.A_cache[-1])
        assert (self.m == Y.shape[1])

        for i in range(self.num_layer - 1, -1, -1):
            dZ = dA * get_activation_de_func(self.activation_func[i])(
                self.Z_cache[i])
            dW = np.dot(dZ, self.A_cache[i].T) / self.m
            db = np.mean(dZ, axis=1, keepdims=True)
            dA = np.dot(self.W[i].T, dZ)
            self.dW_cache[i] = dW
            self.db_cache[i] = db

    def gradient_descent(self, learning_rate):
        for i in range(self.num_layer):
            self.W[i] -= learning_rate * self.dW_cache[i]
            self.b[i] -= learning_rate * self.db_cache[i]

    def save(self, filename: str):
        save_dict = {}
        for i in range(len(self.W)):
            save_dict['W' + str(i)] = self.W[i]
        for i in range(len(self.b)):
            save_dict['b' + str(i)] = self.b[i]
        np.savez(filename, **save_dict)

    def load(self, filename: str):
        params = np.load(filename)
        for i in range(len(self.W)):
            self.W[i] = params['W' + str(i)]
        for i in range(len(self.b)):
            self.b[i] = params['b' + str(i)]


def train(model: BaseRegressionModel,
          X,
          Y,
          step,
          learning_rate,
          print_interval=100,
          test_X=None,
          test_Y=None):
    for s in range(step):
        Y_hat = model.forward(X)
        model.backward(Y)
        model.gradient_descent(learning_rate)
        if s % print_interval == 0:
            loss = model.loss(Y, Y_hat)
            print(f'Step: {s}')
            print(f'Train loss: {loss}')
            if test_X is not None and test_Y is not None:
                accuracy, loss = model.evaluate(test_X,
                                                test_Y,
                                                return_loss=True)
                print(f'Test loss: {loss}')
                print(f'Test accuracy: {accuracy}')
