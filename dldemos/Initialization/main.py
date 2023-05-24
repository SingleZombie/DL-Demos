import abc
from typing import List

import numpy as np

from dldemos.Initialization.points_classification import (generate_plot_set,
                                                          generate_points,
                                                          plot_points,
                                                          visualize)
from dldemos.utils import get_activation_de_func, get_activation_func, sigmoid


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

    def __init__(self,
                 neuron_cnt: List[int],
                 activation_func: List[str],
                 initialization='zeros'):
        assert len(neuron_cnt) - 2 == len(activation_func)
        self.num_layer = len(neuron_cnt) - 1
        self.neuron_cnt = neuron_cnt
        self.activation_func = activation_func
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(self.num_layer):
            if initialization == 'zeros':
                self.W.append(np.zeros((neuron_cnt[i + 1], neuron_cnt[i])))
            elif initialization == 'random':
                self.W.append(
                    np.random.randn(neuron_cnt[i + 1], neuron_cnt[i]) * 5)
            elif initialization == 'he':
                self.W.append(
                    np.random.randn(neuron_cnt[i + 1], neuron_cnt[i]) *
                    np.sqrt(2 / neuron_cnt[i]))
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
            if i == self.num_layer - 1:
                A = sigmoid(Z)
            else:
                A = get_activation_func(self.activation_func[i])(Z)
            if train_mode:
                self.Z_cache[i] = Z
                self.A_cache[i + 1] = A
        return A

    def backward(self, Y):
        assert (self.m == Y.shape[1])

        dA = 0
        for i in range(self.num_layer - 1, -1, -1):
            if i == self.num_layer - 1:
                dZ = self.A_cache[-1] - Y
            else:
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


def main():
    train_X, train_Y = generate_points(400)
    plot_points(train_X, train_Y)
    plot_X = generate_plot_set()

    n_x = train_X.shape[0]
    neuron_list = [n_x, 10, 5, 1]
    activation_list = ['relu', 'relu']
    model1 = DeepNetwork(neuron_list, activation_list, 'zeros')
    model2 = DeepNetwork(neuron_list, activation_list, 'random')
    model3 = DeepNetwork(neuron_list, activation_list, 'he')
    train(model1, train_X, train_Y, 20000, 0.01, 1000)
    train(model2, train_X, train_Y, 20000, 0.01, 1000)
    train(model3, train_X, train_Y, 20000, 0.01, 1000)

    plot_result1 = model1.forward(plot_X, False)
    plot_result2 = model2.forward(plot_X, False)
    plot_result3 = model3.forward(plot_X, False)

    visualize(train_X, train_Y, plot_result1)
    visualize(train_X, train_Y, plot_result2)
    visualize(train_X, train_Y, plot_result3)


if __name__ == '__main__':
    main()
