"""My own plain Numpy implementation of DL basics.
Main equation:
- vector z = w.T * x + b
- matrix Z = W * X + b


Convention:
- weights vector - vertical
- single sample vector - vertical

- Weights matrix W.shape = (n_output, n_input)
"""
import numpy as np


class ActivationFunction:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)


class DenseLayer(ActivationFunction):
    def __init__(self,
                 x,
                 n_output,
                 activation_function=ActivationFunction.sigmoid):

        self.n_input = x.shape[0]               # number of features
        self.n_output = n_output                # number of output neurons
        self.X = x
        self.W = None
        self.Z = None
        self.b = None
        self.activation_function = activation_function

        self.init_weights()
        self.init_bias()

    def init_weights(self):
        if self.W is None:
            self.W = np.random.random((self.n_output, self.n_input))

    def init_bias(self):
        if self.b is None:
            self.b = 1

    def forward_propagation(self):
        self.Z = np.dot(self.W, X) + self.b
        self.Z = self.activation_function(self.Z)


if __name__ == "__main__":
    n = 10
    m = 100
    n_neurons = 4

    X = np.random.random((n, m))

    dense_1 = DenseLayer(X, n_neurons)
    dense_1.forward_propagation()


