import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


def vertical_flip():
    return np.array([[1, 0], [0, -1]])


def rotate(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])


def half_oval(cnt, h=10, w=20):
    x = np.linspace(-w, w, cnt)
    y = np.sqrt(h * h * (1 - x * x / w / w))
    return np.stack((x, y), 1)


def generate_point_set():
    petal1 = half_oval(20)
    petal2 = np.dot(half_oval(20), vertical_flip().T)
    petal = np.concatenate((petal1, petal2), 0)
    petal += [25, 0]
    flower = petal.copy()
    for i in range(5):
        new_petal = np.dot(petal.copy(), rotate(np.radians(60) * (i + 1)).T)
        flower = np.concatenate((flower, new_petal), 0)

    label = np.zeros([40 * 6])
    label[0:40] = 1
    label[40:80] = 1
    label[120:160] = 1

    rng = default_rng()
    noise_indice1 = rng.choice(40 * 6, 10, replace=False)
    label[noise_indice1] = 1 - label[noise_indice1]

    x = flower[:, 0]
    y = flower[:, 1]
    return x, y, label


def generate_plot_set():
    x = np.linspace(-50, 50, 100)
    y = np.linspace(-50, 50, 100)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    return np.stack((xx, yy), axis=1).T


def visualize(X, Y, plot_set_result: np.ndarray):
    x = np.linspace(-50, 50, 100)
    y = np.linspace(-50, 50, 100)
    xx, yy = np.meshgrid(x, y)
    color = plot_set_result.squeeze()
    c = np.where(color < 0.5, 'r', 'g')
    plt.scatter(xx, yy, c=c, marker='s')

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    origin_x = X[:, 0]
    origin_y = X[:, 1]
    origin_color = np.where(Y.squeeze() < 0.5, '#AA0000', '#00AA00')

    plt.scatter(origin_x, origin_y, c=origin_color)

    plt.show()


if __name__ == '__main__':
    x, y, label = generate_point_set()
    c = np.where(label == 0, 'r', 'g')
    plt.scatter(x, y, c=c)

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    plt.show()
