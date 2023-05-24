import abc
import math
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from dldemos.AdvancedOptimizer.optimizer import BaseOptimizer
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
    def get_grad_dict(self) -> Dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def save(self) -> Dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def load(self, state_dict: Dict[str, np.ndarray]):
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
        super().__init__()
        assert len(neuron_cnt) - 1 == len(activation_func)
        self.num_layer = len(neuron_cnt) - 1
        self.neuron_cnt = neuron_cnt
        self.activation_func = activation_func
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(self.num_layer):
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
            A = get_activation_func(self.activation_func[i])(Z)
            if train_mode:
                self.Z_cache[i] = Z
                self.A_cache[i + 1] = A
        return A

    def backward(self, Y):
        # Assume the activation of the lat layer is sigmoid
        assert self.activation_func[-1] == 'sigmoid' and \
            self.neuron_cnt[-1] == 1
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

    def get_grad_dict(self) -> Dict[str, np.ndarray]:
        grad_dict = {}
        for i in range(len(self.dW_cache)):
            grad_dict['W' + str(i)] = self.dW_cache[i]
        for i in range(len(self.db_cache)):
            grad_dict['b' + str(i)] = self.db_cache[i]
        return grad_dict

    def save(self) -> Dict[str, np.ndarray]:
        param_dict = {}
        for i in range(len(self.W)):
            param_dict['W' + str(i)] = self.W[i]
        for i in range(len(self.b)):
            param_dict['b' + str(i)] = self.b[i]
        return param_dict

    def load(self, state_dict: Dict[str, np.ndarray]):
        for i in range(len(self.W)):
            self.W[i] = state_dict['W' + str(i)]
        for i in range(len(self.b)):
            self.b[i] = state_dict['b' + str(i)]


def save_state_dict(model: BaseRegressionModel, optimizer: BaseOptimizer,
                    filename: str):
    state_dict = {'model': model.save(), 'optimizer': optimizer.save()}
    np.savez(filename, **state_dict)


def load_state_dict(model: BaseRegressionModel, optimizer: BaseOptimizer,
                    filename: str):
    state_dict = np.load(filename)
    model.load(state_dict['model'])
    optimizer.load(state_dict['optimizer'])


def train(model: BaseRegressionModel,
          optimizer: BaseOptimizer,
          X,
          Y,
          total_epoch,
          batch_size,
          model_name: str = 'model',
          save_dir: str = 'work_dirs',
          recover_from: Optional[str] = None,
          print_interval: int = 100,
          dev_X=None,
          dev_Y=None,
          plot_mini_batch: bool = False):
    if recover_from:
        load_state_dict(model, optimizer, recover_from)
    m = X.shape[1]
    indices = np.random.permutation(m)
    shuffle_X = X[:, indices]
    shuffle_Y = Y[:, indices]
    num_mini_batch = math.ceil(m / batch_size)
    mini_batch_XYs = []
    for i in range(num_mini_batch):
        if i == num_mini_batch - 1:
            mini_batch_X = shuffle_X[:, i * batch_size:]
            mini_batch_Y = shuffle_Y[:, i * batch_size:]
        else:
            mini_batch_X = shuffle_X[:, i * batch_size:(i + 1) * batch_size]
            mini_batch_Y = shuffle_Y[:, i * batch_size:(i + 1) * batch_size]
        mini_batch_XYs.append((mini_batch_X, mini_batch_Y))
    print(f'Num mini-batch: {num_mini_batch}')

    mini_batch_loss_list = []
    for e in range(total_epoch):
        for mini_batch_X, mini_batch_Y in mini_batch_XYs:
            mini_batch_Y_hat = model.forward(mini_batch_X)
            model.backward(mini_batch_Y)
            optimizer.zero_grad()
            optimizer.add_grad(model.get_grad_dict())
            optimizer.step()
            if plot_mini_batch:
                loss = model.loss(mini_batch_Y, mini_batch_Y_hat)
                mini_batch_loss_list.append(loss)

        currrent_epoch = optimizer.epoch

        if currrent_epoch % print_interval == 0:
            # save_state_dict(
            #     model, optimizer,
            #     os.path.join(save_dir, f'{model_name}_{currrent_epoch}.npz'))
            accuracy, loss = model.evaluate(X, Y, return_loss=True)
            print(f'Epoch: {currrent_epoch}')
            print(f'Train loss: {loss}')
            print(f'Train accuracy: {accuracy}')
            if dev_X is not None and dev_Y is not None:
                accuracy, loss = model.evaluate(dev_X, dev_Y, return_loss=True)
                print(f'Dev loss: {loss}')
                print(f'Dev accuracy: {accuracy}')

        optimizer.increase_epoch()

    save_state_dict(model, optimizer,
                    os.path.join(save_dir, f'{model_name}_latest.npz'))

    if plot_mini_batch:
        plot_length = len(mini_batch_loss_list)
        plot_x = np.linspace(0, plot_length, plot_length)
        plot_y = np.array(mini_batch_loss_list)
        plt.plot(plot_x, plot_y)
        plt.show()
