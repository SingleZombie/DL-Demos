import matplotlib.pyplot as plt
import numpy as np

LLIM = 0
RLIM = 1


def generate_points(cnt):

    PERCENTAGE = 0.2

    X = np.random.rand(int(cnt * (1 - PERCENTAGE)), 2)
    x_1 = X[:, 0]
    x_2 = X[:, 1]
    Y = np.where(x_1 > x_2, 1, 0)

    noise_x = np.random.rand(int(cnt * PERCENTAGE)) / 2
    noise_y = noise_x + np.random.rand(int(cnt * PERCENTAGE)) / 2
    noise_label = np.array([1] * len(noise_x))
    noise_X = np.stack((noise_x, noise_y), axis=1)
    X = np.concatenate((X, noise_X), 0)
    Y = np.concatenate((Y, noise_label), 0)

    return X.T, Y[:, np.newaxis].T


def plot_points(X, Y):
    new_X = X.T
    Y = np.squeeze(Y, 0)
    c = np.where(Y == 0, 'r', 'b')
    new_x = new_X[:, 0]
    new_y = new_X[:, 1]
    plt.scatter(new_x, new_y, color=c)
    plt.show()


def generate_plot_set():
    x = np.linspace(LLIM, RLIM, 100)
    y = np.linspace(LLIM, RLIM, 100)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    return np.stack((xx, yy), axis=1).T


def visualize(X, Y, plot_set_result: np.ndarray):
    x = np.linspace(LLIM, RLIM, 100)
    y = np.linspace(LLIM, RLIM, 100)
    xx, yy = np.meshgrid(x, y)
    color = plot_set_result.squeeze()
    c = np.where(color < 0.5, 'r', 'b')
    plt.scatter(xx, yy, c=c, marker='s')

    plt.xlim(LLIM, RLIM)
    plt.ylim(LLIM, RLIM)

    origin_x = X.T[:, 0]
    origin_y = X.T[:, 1]
    origin_color = np.where(Y.squeeze() < 0.5, '#AA0000', '#0000AA')

    plt.scatter(origin_x, origin_y, c=origin_color)

    plt.show()


def main():
    plot_points(*generate_points(200))


if __name__ == '__main__':
    main()
