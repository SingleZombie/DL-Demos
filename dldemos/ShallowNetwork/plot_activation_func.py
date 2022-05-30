import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def leaky_relu(x):
    return np.maximum(x, 0.1 * x)


x = np.linspace(-3, 3, 100)
y1 = sigmoid(x)
y2 = tanh(x)
y3 = relu(x)
y4 = leaky_relu(x)

plt.subplot(2, 2, 1)
plt.axvline(x=0, color='k')
plt.axhline(y=0, color='k')
plt.plot(x, y1)
plt.title('sigmoid')

plt.subplot(2, 2, 2)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.plot(x, y2)
plt.title('tanh')

plt.subplot(2, 2, 3)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.plot(x, y3)
plt.title('relu')

plt.subplot(2, 2, 4)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.plot(x, y4)
plt.title('leaky_relu')

plt.show()
