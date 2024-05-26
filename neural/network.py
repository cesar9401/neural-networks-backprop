import numpy as np
from neural.functions import sigmoid, sigmoid_prime, step_function, hyperbolic_tangent, hyperbolic_tangent_prime, \
    identity_function, step_function_prime, identity_function_prime
import random


class Network:
    def __init__(self, sizes, hidden_activation_func: str, output_activation_func: str):
        self.layer_sizes = sizes
        self.layers = len(self.layer_sizes)

        self.biases = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]  # from index 1 to self.sizes - 1
        # to create a matrix of (i) x (i + 1), add a matrix of (i) x (i + 1) with random values
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

        self.hidden_activation, self.hidden_activation_prime = self.get_activation_function(hidden_activation_func)
        self.output_activation, self.output_activation_prime = self.get_activation_function(output_activation_func)

    def feedforward(self, activation):  # activation of any layer or input of the first layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # matrix operation (w x a) + b
            activation = self.hidden_activation(z)
        return self.output_activation(z)

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                accuracy = self.evaluate(test_data)
                print(f'Epoch {epoch + 1}: {accuracy} / {len(test_data)}')
            else:
                print(f'Epoch {epoch + 1} complete')

    def update_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(w - learning_rate * nw) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [(b - learning_rate * nb) for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.hidden_activation(z)
            activations.append(activation)
        activations[-1] = self.output_activation(z)

        delta = self.cost_prime(activations[-1], y) * self.output_activation_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.layers):
            z = zs[-layer]
            sp = self.hidden_activation_prime(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    @staticmethod
    def cost_prime(actual, predicted):
        return actual - predicted

    @staticmethod
    def get_activation_function(name):
        if name == 'sigmoid':
            return sigmoid, sigmoid_prime
        elif name == 'tanh':
            return hyperbolic_tangent, hyperbolic_tangent_prime
        elif name == 'step':
            return step_function, step_function_prime
        elif name == 'identity':
            return identity_function, identity_function_prime
