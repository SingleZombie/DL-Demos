import matplotlib.pyplot as plt
import numpy as np

LLIM = 0
RLIM = 1


def generate_points(cnt):
    x = np.random.rand(cnt)
    y = np.random.rand(cnt)
    X = np.stack([x, y], 1)
    Y = np.where(y > x * x, np.where(y > x**0.5, 0, 1), 2)
    return X.T, Y[..., np.newaxis].T


def plot_points(X, Y):
    new_X = X.T
    Y = np.squeeze(Y, 0)
    color_map = np.array(['r', 'g', 'b'])
    c = color_map[Y]
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
    color_map_1 = np.array(['r', 'g', 'b'])
    color_map_2 = ['#AA0000', '#00AA00', '#0000AA']

    c = color_map_1[color]
    plt.scatter(xx, yy, c=c, marker='s')

    plt.xlim(LLIM, RLIM)
    plt.ylim(LLIM, RLIM)

    origin_x = X.T[:, 0]
    origin_y = X.T[:, 1]
    origin_color = Y.squeeze(0)
    origin_color = [color_map_2[oc] for oc in origin_color]

    plt.scatter(origin_x, origin_y, c=origin_color)

    plt.show()


def main():
    plot_points(*generate_points(400))


if __name__ == '__main__':
    main()
