import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def hyperbolic_tangent(z):
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)


def hyperbolic_tangent_prime(z):
    return pow((2 * np.exp(z)) / (np.exp(2 * z) + 1), 2)


def identity_function(z):
    return z


def identity_function_prime(z):
    return 1


def step_function(z):
    return 1 if z >= 0 else 0


def step_function_prime(z):
    return 0
