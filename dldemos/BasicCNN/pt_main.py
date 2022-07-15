import math

import numpy as np
import torch
import torch.nn as nn

from dldemos.BasicCNN.dataset import get_cat_set


def init_model(device='cpu'):
    model = nn.Sequential(nn.Conv2d(3, 16, 11, 3), nn.BatchNorm2d(16),
                          nn.ReLU(True), nn.MaxPool2d(2, 2),
                          nn.Conv2d(16, 32, 5), nn.BatchNorm2d(32),
                          nn.ReLU(True), nn.MaxPool2d(2, 2),
                          nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                          nn.ReLU(True), nn.Conv2d(64, 64, 3),
                          nn.BatchNorm2d(64), nn.ReLU(True),
                          nn.MaxPool2d(2, 2), nn.Flatten(),
                          nn.Linear(3136, 2048), nn.ReLU(True),
                          nn.Linear(2048, 1), nn.Sigmoid()).to(device)

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0)

    model.apply(weights_init)

    print(model)
    return model


def train(model: nn.Module,
          train_X: np.ndarray,
          train_Y: np.ndarray,
          optimizer: torch.optim.Optimizer,
          loss_fn: nn.Module,
          batch_size: int,
          num_epoch: int,
          device: str = 'cpu'):
    m = train_X.shape[0]
    indices = np.random.permutation(m)
    shuffle_X = train_X[indices, ...]
    shuffle_Y = train_Y[indices, ...]
    num_mini_batch = math.ceil(m / batch_size)
    mini_batch_XYs = []
    for i in range(num_mini_batch):
        if i == num_mini_batch - 1:
            mini_batch_X = shuffle_X[i * batch_size:, ...]
            mini_batch_Y = shuffle_Y[i * batch_size:, ...]
        else:
            mini_batch_X = shuffle_X[i * batch_size:(i + 1) * batch_size, ...]
            mini_batch_Y = shuffle_Y[i * batch_size:(i + 1) * batch_size, ...]
        mini_batch_X = torch.from_numpy(mini_batch_X)
        mini_batch_Y = torch.from_numpy(mini_batch_Y).float()
        mini_batch_XYs.append((mini_batch_X, mini_batch_Y))
    print(f'Num mini-batch: {num_mini_batch}')

    for e in range(num_epoch):
        for mini_batch_X, mini_batch_Y in mini_batch_XYs:
            mini_batch_X = mini_batch_X.to(device)
            mini_batch_Y = mini_batch_Y.to(device)
            mini_batch_Y_hat = model(mini_batch_X)
            loss: torch.Tensor = loss_fn(mini_batch_Y_hat, mini_batch_Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {e}. loss: {loss}')


def evaluate(model: nn.Module,
             test_X: np.ndarray,
             test_Y: np.ndarray,
             device='cpu'):
    test_X = torch.from_numpy(test_X).to(device)
    test_Y = torch.from_numpy(test_Y).to(device)
    test_Y_hat = model(test_X)
    predicts = torch.where(test_Y_hat > 0.5, 1, 0)
    score = torch.where(predicts == test_Y, 1.0, 0.0)
    acc = torch.mean(score)
    print(f'Accuracy: {acc}')


def main():
    train_X, train_Y, test_X, test_Y = get_cat_set(
        'dldemos/LogisticRegression/data/archive/dataset',
        train_size=1500,
        format='nchw')
    print(train_X.shape)  # (m, 3, 224, 224)
    print(train_Y.shape)  # (m, 1)

    device = 'cuda:0'
    num_epoch = 20
    batch_size = 16
    model = init_model(device)
    optimizer = torch.optim.Adam(model.parameters(), 5e-4)
    loss_fn = torch.nn.BCELoss()
    train(model, train_X, train_Y, optimizer, loss_fn, batch_size, num_epoch,
          device)
    evaluate(model, test_X, test_Y, device)


if __name__ == '__main__':
    main()
