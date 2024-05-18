import pandas as pd
import numpy as np
import re

from network import Network

df = pd.read_csv('data/demo.csv')
regex_input = 'i[1-9][0-9]*'
regex_output = 'o[1-9][0-9]*'

in_cols = [col for col in df.columns if re.search(regex_input, col)]
out_cols = [col for col in df.columns if re.search(regex_output, col)]

total_inputs = []
total_outputs = []

for index, row in df.iterrows():
    inputs = []
    for col in in_cols:
        inputs.append(row[col])
    total_inputs.append(inputs)

    outputs = []
    for col in out_cols:
        outputs.append(row[col])
    total_outputs.append(outputs)

training_input = np.array(total_inputs)
training_output = np.array(total_outputs)

training_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(training_input, training_output)]

network = Network([5, 50, 60, 1], 'sigmoid', 'step')
network.SGD(list(training_data), 1000, 1, 0.4, None)

for x, y in training_data:
    output = network.feedforward(x)
    expected = y
    print(f'(result {expected == output})\t\texpected: {expected} -> output: {output}')
