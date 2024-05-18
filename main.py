from PyQt5.QtWidgets import QApplication, QStyleFactory
import sys
import re

from network import Network
from ui import main_widget

import pandas as pd
import numpy as np

REGEX_INPUT = 'i[1-9][0-9]*'
REGEX_OUTPUT = 'o[1-9][0-9]*'


class Main:
    def __init__(self):
        self.network: Network | None = None

        app = QApplication(sys.argv)
        app.aboutToQuit.connect(app.deleteLater)
        app.setStyle(QStyleFactory.create('gtk'))
        self.window = main_widget.MainWidget(self)
        self.window.show()
        sys.exit(app.exec_())

    def start_training(self, layer_sizes, training_file, hidden_function, output_function, epochs, learning_rate):
        training_data = self.read_file(training_file)

        self.network = Network(layer_sizes, hidden_function, output_function)
        self.network.SGD(training_data, epochs, 1, learning_rate, None)

    def start_testing(self, testing_file):
        if self.network is None:
            return

        testing_data = self.read_file(testing_file)
        for x, expected in testing_data:
            output = self.network.feedforward(x)
            print(f'(result {expected == output})\t\texpected: {expected} -> output: {output}')

    def read_file(self, filename):
        df, in_cols, out_cols = self.get_total_inputs_and_outputs(filename)

        total_inputs, total_outputs = [], []
        for index, row in df.iterrows():
            inputs = []
            for col in in_cols:
                inputs.append(row[col])
            total_inputs.append(inputs)

            outputs = []
            for col in out_cols:
                outputs.append(row[col])
            total_outputs.append(outputs)

        file_input = np.array(total_inputs)
        file_output = np.array(total_outputs)
        return [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(file_input, file_output)]

    @staticmethod
    def get_total_inputs_and_outputs(filename):
        df = pd.read_csv(filename)

        in_cols = [col for col in df.columns if re.search(REGEX_INPUT, col)]
        out_cols = [col for col in df.columns if re.search(REGEX_OUTPUT, col)]

        return df, in_cols, out_cols


# from network import Network
# import numpy as np
#
# print('Hello, there!')
#
#
# def logical_or(x):
#     return int(x[0] or x[1])
#
#
# def logical_and(x):
#     return int(x[0] and x[1])
#
#
# # Generate inputs for the logical OR function
# inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# inputs1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
#
# # Generate outputs for the logical OR function
# outputs = np.array([logical_or(x) for x in inputs])
# test_output = np.array([[0], [1], [1], [0]])
#
# # Combine inputs and outputs into a list of 2-tuples
# training_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(inputs, outputs)]
# test_data = [(x.reshape(-1, 1), y.reshape(-1, 1)) for x, y in zip(inputs1, test_output)]
#
# network1 = Network([2, 100, 200, 300, 200, 100, 1], 'sigmoid', 'step')
# network1.SGD(test_data, 1000, 1, 0.2, None)
#
# for x, y in test_data:
#     output = network1.feedforward(x)
#     expected = y
#     # expected = np.argmax(y)
#     print(f'(result {expected == output})\t\texpected: {expected} -> output: {output}')
#     # print(y[0][0] == network1.feedforward(x))
#
# # result = network1.evaluate(test_data)
# # print(result)

if __name__ == '__main__':
    main = Main()
