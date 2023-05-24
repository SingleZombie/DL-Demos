from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from dldemos.MulticlassClassification.points_classification import (
    generate_plot_set, generate_points, plot_points, visualize)


class MulticlassClassificationNet():

    def __init__(self, neuron_cnt: List[int]):
        self.num_layer = len(neuron_cnt) - 1
        self.neuron_cnt = neuron_cnt
        self.W = []
        self.b = []
        for i in range(self.num_layer):
            new_W = torch.empty(neuron_cnt[i + 1], neuron_cnt[i])
            new_b = torch.empty(neuron_cnt[i + 1], 1)
            torch.nn.init.kaiming_normal_(new_W, nonlinearity='relu')
            torch.nn.init.kaiming_normal_(new_b, nonlinearity='relu')
            self.W.append(torch.nn.Parameter(new_W))
            self.b.append(torch.nn.Parameter(new_b))
        self.trainable_vars = self.W + self.b
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        A = X
        for i in range(self.num_layer):
            Z = torch.matmul(self.W[i], A) + self.b[i]
            if i == self.num_layer - 1:
                A = F.softmax(Z, 0)
            else:
                A = F.relu(Z)

        return A

    def loss(self, Y, Y_hat):
        return self.loss_fn(Y_hat.T, Y)

    def evaluate(self, X, Y, return_loss=False):
        Y_hat = self.forward(X)
        Y_predict = Y
        Y_hat_predict = torch.argmax(Y_hat, 0)
        res = (Y_predict == Y_hat_predict).float()
        accuracy = torch.mean(res)
        if return_loss:
            loss = self.loss(Y, Y_hat)
            return accuracy, loss
        else:
            return accuracy


def train(model: MulticlassClassificationNet,
          X,
          Y,
          step,
          learning_rate,
          print_interval=100):
    optimizer = torch.optim.Adam(model.trainable_vars, learning_rate)
    for s in range(step):
        Y_hat = model.forward(X)
        cost = model.loss(Y, Y_hat)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if s % print_interval == 0:
            accuracy, loss = model.evaluate(X, Y, return_loss=True)
            print(f'Step: {s}')
            print(f'Accuracy: {accuracy}')
            print(f'Train loss: {loss}')


def main():
    train_X, train_Y = generate_points(400)
    plot_points(train_X, train_Y)
    plot_X = generate_plot_set()

    # X: [2, m]
    # Y: [1, m]

    train_X_pt = torch.tensor(train_X, dtype=torch.float32)
    train_Y_pt = torch.tensor(train_Y.squeeze(0), dtype=torch.long)

    print(train_X_pt.shape)
    print(train_Y_pt.shape)

    # X: [2, m]
    # Y: [m]

    n_x = 2
    neuron_list = [n_x, 10, 10, 3]
    model = MulticlassClassificationNet(neuron_list)
    train(model, train_X_pt, train_Y_pt, 5000, 0.001, 1000)

    plot_result = model.forward(torch.Tensor(plot_X))
    plot_result = torch.argmax(plot_result, 0).numpy()
    plot_result = np.expand_dims(plot_result, 0)

    visualize(train_X, train_Y, plot_result)


if __name__ == '__main__':
    main()
