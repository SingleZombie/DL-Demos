import matplotlib.pyplot as plt
import numpy as np


def generate_points(cnt):

    def draw_ring(cnt, inner_radius, outer_radius):
        angle_arr = np.random.rand(cnt) * np.pi * 2
        length_arr = np.random.rand(cnt) * (outer_radius -
                                            inner_radius) + inner_radius
        return length_arr * np.cos(angle_arr), length_arr * np.sin(angle_arr)

    red_cnt = cnt // 2
    blue_cnt = cnt - red_cnt

    red_x, red_y = draw_ring(red_cnt, 5, 6)
    blue_x, blue_y = draw_ring(blue_cnt, 6, 7)
    X = np.stack((np.concatenate(
        (red_x, blue_x)), np.concatenate((red_y, blue_y))), 1)
    Y = np.array([0] * red_cnt + [1] * blue_cnt)
    return X.T, Y[..., np.newaxis].T


def plot_points(X, Y):
    new_X = X.T
    Y = np.squeeze(Y, 0)
    c = np.where(Y == 0, 'r', 'b')
    new_x = new_X[:, 0]
    new_y = new_X[:, 1]
    plt.scatter(new_x, new_y, color=c)
    plt.show()


def generate_plot_set():
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    return np.stack((xx, yy), axis=1).T


def visualize(X, Y, plot_set_result: np.ndarray):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    xx, yy = np.meshgrid(x, y)
    color = plot_set_result.squeeze()
    c = np.where(color < 0.5, 'r', 'b')
    plt.scatter(xx, yy, c=c, marker='s')

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    origin_x = X.T[:, 0]
    origin_y = X.T[:, 1]
    origin_color = np.where(Y.squeeze() < 0.5, '#AA0000', '#0000AA')

    plt.scatter(origin_x, origin_y, c=origin_color)

    plt.show()


def main():
    plot_points(*generate_points(400))


if __name__ == '__main__':
    main()
