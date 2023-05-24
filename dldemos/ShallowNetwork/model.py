import abc

import numpy as np

from dldemos.utils import relu, relu_de, sigmoid


class BaseRegressionModel(metaclass=abc.ABCMeta):
    # Use Cross Entropy as the cost function

    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, X, train_mode=True):
        # if self.train_mode:
        # forward_train()
        # else:
        # forward_test()
        pass

    @abc.abstractmethod
    def backward(self, Y):
        pass

    @abc.abstractmethod
    def gradient_descent(self, learning_rate=0.001):
        pass

    def loss(self, Y_hat, Y):
        return np.mean(-(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)))

    def evaluate(self, X, Y):
        Y_hat = self.forward(X, train_mode=False)
        predicts = np.where(Y_hat > 0.5, 1, 0)
        score = np.mean(np.where(predicts == Y, 1, 0))
        print(f'Accuracy: {score}')


class LogisticRegression(BaseRegressionModel):

    def __init__(self, n_x):
        super().__init__()
        self.n_x = n_x
        self.w = np.zeros((n_x, 1))
        self.b = 0

    def forward(self, X, train_mode=True):
        Z = np.dot(self.w.T, X) + self.b
        A = sigmoid(Z)  # hat_Y = A
        if train_mode:
            self.m_cache = X.shape[1]
            self.X_cache = X
            self.A_cache = A
        return A

    def backward(self, Y):
        d_Z = self.A_cache - Y
        d_w = np.dot(self.X_cache, d_Z.T) / self.m_cache
        d_b = np.mean(d_Z)
        self.d_w_cache = d_w
        self.d_b_cache = d_b

    def gradient_descent(self, learning_rate=0.001):
        self.w -= learning_rate * self.d_w_cache
        self.b -= learning_rate * self.d_b_cache


class ShallowNetwork(BaseRegressionModel):
    # x -> hidden layer -> output layer -> y
    # hidden layer (n_1 relu)
    # output layer (1 sigmoid)
    def __init__(self, n_x, n_1):
        super().__init__()
        self.n_x = n_x
        self.n_1 = n_1
        self.W1 = np.random.randn(n_1, n_x) * 0.01
        self.b1 = np.zeros((n_1, 1))
        self.W2 = np.random.randn(1, n_1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, X, train_mode=True):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)
        if train_mode:
            self.m_cache = X.shape[1]
            self.X_cache = X
            self.Z1_cache = Z1
            self.A1_cache = A1
            self.A2_cache = A2
        return A2

    def backward(self, Y):
        dZ2 = self.A2_cache - Y
        dW2 = np.dot(dZ2, self.A1_cache.T) / self.m_cache
        db2 = np.sum(dZ2, axis=1, keepdims=True) / self.m_cache
        dA1 = np.dot(self.W2.T, dZ2)

        dZ1 = dA1 * relu_de(self.Z1_cache)
        dW1 = np.dot(dZ1, self.X_cache.T) / self.m_cache
        db1 = np.sum(dZ1, axis=1, keepdims=True) / self.m_cache

        self.dW2_cache = dW2
        self.dW1_cache = dW1
        self.db2_cache = db2
        self.db1_cache = db1

    def gradient_descent(self, learning_rate=0.001):
        self.W1 -= learning_rate * self.dW1_cache
        self.b1 -= learning_rate * self.db1_cache
        self.W2 -= learning_rate * self.dW2_cache
        self.b2 -= learning_rate * self.db2_cache


def train_model(model: BaseRegressionModel,
                X_train,
                Y_train,
                X_test,
                Y_test,
                steps=1000,
                learning_rate=0.001,
                print_interval=100):
    for step in range(steps):
        Y_hat = model.forward(X_train)
        model.backward(Y_train)
        model.gradient_descent(learning_rate)
        if step % print_interval == 0:
            train_loss = model.loss(Y_hat, Y_train)
            print(f'Step {step}')
            print(f'Train loss: {train_loss}')
            model.evaluate(X_test, Y_test)
