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
    """"Collection of activation functions"""
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def tanh(z):
        return np.tanh(z)

class LossFunction:
    @staticmethod
    def mse(y_pred, y_true):
        if y_pred.shape == y_true.shape:
            pred_len = y_pred.shape[0]
            return (1/pred_len) * np.sum(np.square(y_pred - y_true))
        else:
            raise Exception(f'Vectors are different shape: {y_pred.shape} and {y_true.shape}')

class Layer:
    pass


class DenseLayer(ActivationFunction):
    # TODO: Dense Layer should inherit after more general 'Layer' class
    def __init__(self,
                 x,
                 units,
                 activation_function=ActivationFunction.sigmoid):

        self.n_input = x.shape[0]               # number of features
        self.units = units                # number of output neurons
        self.X = x
        self.W = None
        self.Z = None
        self.b = None
        self.activation_function = activation_function

        self.init_weights()
        self.init_bias()

    def init_weights(self):
        if self.W is None:
            self.W = np.random.random((self.units, self.n_input))

    def init_bias(self):
        if self.b is None:
            self.b = 1

    def forward_propagation(self):
        self.Z = np.dot(self.W, X) + self.b
        self.Z = self.activation_function(self.Z)


class Model(LossFunction):
    """Class represents whole model"""
    def __init__(self, x, y, n_epochs):
        self.x = x
        self.y = y
        self.n_epochs = n_epochs
        self.model = None
        self.pred = None
        self.loss = None

    def add(self, layer):
        self.model = layer

    def compute_prediction(self):
        self.model.forward_propagation()

    def calulate_loss(self):
        self.loss = LossFunction.mse(self.model.Z, self.y)


if __name__ == "__main__":
    n = 10                            # number of features
    m = 1                             # number of samples
    n_neurons = 1
    n_epochs = 2

    X = np.random.random((n, m))        # random input matrix
    Y = np.random.random((1, m))        # random output matrix

    # dense_1 = DenseLayer(X, units=n_neurons)
    # dense_1.forward_propagation()
    # print(dense_1.Z.shape)

    model = Model(X, Y, 1)
    model.add(DenseLayer(X, units=n_neurons))

    model.compute_prediction()

    model.calulate_loss()

    print(model.loss)

