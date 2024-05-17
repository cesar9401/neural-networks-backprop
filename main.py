from network import Network
import numpy as np

print('Hello, there!')


def logical_or(x):
    return int(x[0] or x[1])


def logical_and(x):
    return int(x[0] and x[1])


# Generate inputs for the logical OR function
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
inputs1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Generate outputs for the logical OR function
outputs = np.array([logical_or(x) for x in inputs])
test_output = np.array([[0], [1], [1], [0]])

# Combine inputs and outputs into a list of 2-tuples
training_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(inputs, outputs)]
test_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(inputs1, test_output)]

network1 = Network([2, 100, 200, 300, 200, 100, 1], 'tanh', 'step')
network1.SGD(test_data, 1000, 1, 0.2, None)

for x, y in test_data:
    output = network1.feedforward(x)
    expected = np.argmax(y)
    print(f'(result {y[0][0] == output})\t\texpected: {y[0][expected]} -> output: {output}')
    # print(y[0][0] == network1.feedforward(x))

# result = network1.evaluate(test_data)
# print(result)
