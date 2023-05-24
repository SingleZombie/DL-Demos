from typing import List

import numpy as np
import tensorflow as tf

from dldemos.MulticlassClassification.points_classification import (
    generate_plot_set, generate_points, plot_points, visualize)


class MulticlassClassificationNet():

    def __init__(self, neuron_cnt: List[int]):
        self.num_layer = len(neuron_cnt) - 1
        self.neuron_cnt = neuron_cnt
        self.W = []
        self.b = []
        initializer = tf.keras.initializers.HeNormal(seed=1)
        for i in range(self.num_layer):
            self.W.append(
                tf.Variable(
                    initializer(shape=(neuron_cnt[i + 1], neuron_cnt[i]))))
            self.b.append(
                tf.Variable(initializer(shape=(neuron_cnt[i + 1], 1))))
        self.trainable_vars = self.W + self.b

    def forward(self, X):
        A = X
        for i in range(self.num_layer):
            Z = tf.matmul(self.W[i], A) + self.b[i]
            if i == self.num_layer - 1:
                A = tf.keras.activations.softmax(Z)
            else:
                A = tf.keras.activations.relu(Z)

        return A

    def loss(self, Y, Y_hat):
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(tf.transpose(Y),
                                                     tf.transpose(Y_hat)))

    def evaluate(self, X, Y, return_loss=False):
        Y_hat = self.forward(X)
        Y_predict = tf.argmax(Y, 0)
        Y_hat_predict = tf.argmax(Y_hat, 0)
        res = tf.cast(Y_predict == Y_hat_predict, tf.float32)
        accuracy = tf.reduce_mean(res)
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
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    for s in range(step):
        with tf.GradientTape() as tape:
            Y_hat = model.forward(X)
            cost = model.loss(Y, Y_hat)
        grads = tape.gradient(cost, model.trainable_vars)
        optimizer.apply_gradients(zip(grads, model.trainable_vars))
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

    train_X_tf = tf.constant(train_X, dtype=tf.float32)
    train_Y_tf = tf.transpose(tf.one_hot(train_Y.squeeze(0), 3))

    # X: [2, m]
    # Y: [3, m]

    n_x = 2
    neuron_list = [n_x, 10, 10, 3]
    model = MulticlassClassificationNet(neuron_list)
    train(model, train_X_tf, train_Y_tf, 5000, 0.001, 1000)

    plot_result = model.forward(plot_X)
    plot_result = tf.argmax(plot_result, 0).numpy()
    plot_result = np.expand_dims(plot_result, 0)

    visualize(train_X, train_Y, plot_result)


if __name__ == '__main__':
    main()
