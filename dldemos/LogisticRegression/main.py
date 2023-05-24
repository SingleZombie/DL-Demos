import os.path as osp
from glob import glob
from random import shuffle

import cv2
import numpy as np


def generate_data(dir='data/archive/dataset', input_shape=(224, 224)):

    def load_dataset(dir, data_num):
        cat_images = glob(osp.join(dir, 'cats', '*.jpg'))
        dog_images = glob(osp.join(dir, 'dogs', '*.jpg'))
        cat_tensor = []
        dog_tensor = []

        for idx, image in enumerate(cat_images):
            if idx >= data_num:
                break
            i = cv2.imread(image) / 255
            i = cv2.resize(i, input_shape)
            cat_tensor.append(i)

        for idx, image in enumerate(dog_images):
            if idx >= data_num:
                break
            i = cv2.imread(image) / 255
            i = cv2.resize(i, input_shape)
            dog_tensor.append(i)

        X = cat_tensor + dog_tensor
        Y = [1] * len(cat_tensor) + [0] * len(dog_tensor)
        X_Y = list(zip(X, Y))
        shuffle(X_Y)
        X, Y = zip(*X_Y)
        return X, Y

    train_X, train_Y = load_dataset(osp.join(dir, 'training_set'), 400)
    test_X, test_Y = load_dataset(osp.join(dir, 'test_set'), 100)
    return train_X, train_Y, test_X, test_Y


def resize_input(a: np.ndarray):
    h, w, c = a.shape
    a.resize((h * w * c))
    return a


def init_weights(n_x=224 * 224 * 3):
    w = np.zeros((n_x, 1))
    b = 0.0
    return w, b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(w, b, X):
    return sigmoid(np.dot(w.T, X) + b)


def loss(y_hat, y):
    return np.mean(-(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


def train_step(w, b, X, Y, lr):
    m = X.shape[1]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    d_Z = A - Y
    d_w = np.dot(X, d_Z.T) / m
    d_b = np.mean(d_Z)
    return w - lr * d_w, b - lr * d_b


def train(train_X, train_Y, step=1000, learning_rate=0.00001):
    w, b = init_weights()
    print(f'learning rate: {learning_rate}')
    for i in range(step):
        w, b = train_step(w, b, train_X, train_Y, learning_rate)
        if i % 10 == 0:
            y_hat = predict(w, b, train_X)
            ls = loss(y_hat, train_Y)
            print(f'step {i} loss: {ls}')
    return w, b


def test(w, b, test_X, test_Y):
    y_hat = predict(w, b, test_X)
    predicts = np.where(y_hat > 0.5, 1, 0)
    score = np.mean(np.where(predicts == test_Y, 1, 0))
    print(f'Accuracy: {score}')


def main():
    train_X, train_Y, test_X, test_Y = generate_data()

    train_X = [resize_input(x) for x in train_X]
    test_X = [resize_input(x) for x in test_X]
    train_X = np.array(train_X).T
    train_Y = np.array(train_Y)
    train_Y = train_Y.reshape((1, -1))
    test_X = np.array(test_X).T
    test_Y = np.array(test_Y)
    test_Y = test_Y.reshape((1, -1))
    print(f'Training set size: {train_X.shape[1]}')
    print(f'Test set size: {test_X.shape[1]}')

    w, b = train(train_X, train_Y, learning_rate=0.0002)

    test(w, b, test_X, test_Y)


if __name__ == '__main__':
    main()
