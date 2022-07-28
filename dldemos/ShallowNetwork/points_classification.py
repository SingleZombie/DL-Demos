import numpy as np

from dldemos.ShallowNetwork.genereate_points import (generate_plot_set,
                                                     generate_point_set,
                                                     visualize)
from dldemos.ShallowNetwork.model import (LogisticRegression, ShallowNetwork,
                                          train_model)


def main():
    x, y, label = generate_point_set()
    # x: [240]
    # y: [240]
    # label: [240]

    X = np.stack((x, y), axis=1)
    Y = np.expand_dims(label, axis=1)
    # X: [240, 2]
    # Y: [240, 1]

    indices = np.random.permutation(X.shape[0])
    X_train = X[indices[0:200], :].T
    Y_train = Y[indices[0:200], :].T
    X_test = X[indices[200:], :].T
    Y_test = Y[indices[200:], :].T
    # X_train: [2, 200]
    # Y_train: [1, 200]
    # X_test: [2, 40]
    # Y_test: [1, 40]

    n_x = 2

    model1 = LogisticRegression(n_x)
    model2 = ShallowNetwork(n_x, 2)
    model3 = ShallowNetwork(n_x, 4)
    model4 = ShallowNetwork(n_x, 10)
    train_model(model1, X_train, Y_train, X_test, Y_test, 500, 0.0001, 50)
    train_model(model2, X_train, Y_train, X_test, Y_test, 2000, 0.01, 100)
    train_model(model3, X_train, Y_train, X_test, Y_test, 5000, 0.01, 500)
    train_model(model4, X_train, Y_train, X_test, Y_test, 5000, 0.01, 500)

    visualize_X = generate_plot_set()
    plot_result = model4.forward(visualize_X, train_mode=False)
    visualize(X, Y, plot_result)


if __name__ == '__main__':
    main()
