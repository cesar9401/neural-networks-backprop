import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def hyperbolic_tangent(z):
    return np.tanh(z)


def hyperbolic_tangent_prime(z):
    return 1.0 - np.tanh(z) ** 2


def identity_function(z):
    return z


def identity_function_prime(z):
    return np.ones_like(z)


def step_function(z):
    return np.where(z > 0, 1, 0)


def step_function_prime(z):
    return np.where(z > 0, 0, 0)
