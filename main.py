from PyQt5.QtWidgets import QApplication, QStyleFactory
import sys
import re

from neural.network import Network
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
        print(f'Trained with: {training_file}')

    def start_testing(self, testing_file):
        if self.network is None:
            return

        testing_data = self.read_file(testing_file)
        correct = 0

        df, in_cols, out_cols = self.get_total_inputs_and_outputs(testing_file)

        for file_data, df_data in zip(testing_data, df.iterrows()):
            output = self.network.feedforward(file_data[0])
            inputs = [df_data[1][col] for col in in_cols]
            outputs = [df_data[1][col] for col in out_cols]
            result = (file_data[1] == output)
            correct = correct + 1 if result[0, 0] else correct
            print(f'(result: {result[0, 0]})\t\tinput: {inputs}, expected: {outputs}, output: {output.ravel()}')

        accuracy = (correct / len(testing_data)) * 100
        print(f'Accuracy: {accuracy}%')
        print(f'Tested: {testing_file}')

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

    def get_total_inputs_and_outputs(self, filename):
        df = self.get_df(filename)

        in_cols = [col for col in df.columns if re.search(REGEX_INPUT, col)]
        out_cols = [col for col in df.columns if re.search(REGEX_OUTPUT, col)]

        return df, in_cols, out_cols

    @staticmethod
    def get_df(filename):
        return pd.read_csv(filename)


if __name__ == '__main__':
    main = Main()
