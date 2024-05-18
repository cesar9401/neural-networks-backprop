from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QGridLayout, QVBoxLayout, QGroupBox, QPushButton, QComboBox, \
    QLabel, QFileDialog, QDialog, QSpinBox, QHBoxLayout, QDoubleSpinBox


class MainWidget(QWidget):
    def __init__(self, main):
        super().__init__()

        self.main = main

        self.functions0 = ['sigmoid', 'tanh', 'step', 'identity']
        self.functions1 = ['sigmoid', 'tanh', 'step', 'identity']
        self.layers = []

        self.setGeometry(100, 100, 800, 600)
        self.set_center()
        self.setWindowTitle("IA")

        self.training_data: str | None = None
        self.test_data: str | None = None

        self.grid = QGridLayout()
        self.setLayout(self.grid)

        self.left_layout = QVBoxLayout()
        vertical_group = QGroupBox()
        self.left_layout.addWidget(vertical_group)

        self.right_layout = QVBoxLayout()
        vertical_right_group = QGroupBox()
        self.right_layout.addWidget(vertical_right_group)

        # left
        layout = QVBoxLayout()
        layout.setSpacing(10)
        vertical_group.setLayout(layout)

        # right
        self.layout_right = QVBoxLayout()
        self.layout_right.setAlignment(Qt.AlignTop)
        self.layout_right.setSpacing(10)
        vertical_right_group.setLayout(self.layout_right)

        # open file
        self.open_training_btn = QPushButton("Open training data")
        self.open_training_btn.clicked.connect(self.open_training_data)
        layout.addWidget(self.open_training_btn)

        # add hidden layer test
        self.add_hidden_btn = QPushButton("Add hidden layer")
        self.add_hidden_btn.clicked.connect(self.add_hidden_layer)
        layout.addWidget(self.add_hidden_btn)

        # functions for hidden layer
        self.hidden_function_options = QComboBox()
        self.hidden_function_options.addItems(self.functions0)
        self.hidden_function_options.setCurrentIndex(0)
        layout.addWidget(QLabel('Function for hidden layer'))
        layout.addWidget(self.hidden_function_options)

        # functions for output layer
        self.output_function_options = QComboBox()
        self.output_function_options.addItems(self.functions1)
        self.output_function_options.setCurrentIndex(2)
        layout.addWidget(QLabel('Function for output layer'))
        layout.addWidget(self.output_function_options)

        # epochs
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2147483647)
        self.epochs_spin.setValue(1000)
        layout.addWidget(QLabel('Epochs'))
        layout.addWidget(self.epochs_spin)

        # learning rate
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.00000001, 2147483647)
        self.learning_rate_spin.setValue(0.2)
        layout.addWidget(QLabel('Learning rate'))
        layout.addWidget(self.learning_rate_spin)

        # start training
        self.start_training_btn = QPushButton("Start training")
        self.start_training_btn.clicked.connect(self.start_training)
        layout.addWidget(self.start_training_btn)

        # open test file
        self.open_test_btn = QPushButton("Open test data")
        self.open_test_btn.clicked.connect(self.open_test_data)
        layout.addWidget(self.open_test_btn)

        # start test
        self.start_test_btn = QPushButton("Start test")
        self.start_test_btn.clicked.connect(self.start_test)
        layout.addWidget(self.start_test_btn)

        self.grid.addLayout(self.left_layout, 0, 0)
        self.grid.addLayout(self.right_layout, 0, 1, 9, 9)

    def open_training_data(self):
        self.training_data = None
        self.layers = []
        dlg = QFileDialog()
        dlg.setAcceptMode(QFileDialog.AcceptOpen)
        dlg.setFileMode(QFileDialog.ExistingFile)

        if dlg.exec_() != QDialog.Accepted:
            return

        file_name = dlg.selectedFiles()[0]
        if not file_name:
            return

        self.training_data = file_name
        print(self.training_data)

        df, cols_input, cols_output = self.main.get_total_inputs_and_outputs(self.training_data)
        print(cols_input, cols_output)
        self.layers = []
        self.layers = [len(cols_input), len(cols_output)]
        self.print_layers()

    def start_training(self):
        if self.training_data is None:
            return

        hidden_function = self.hidden_function_options.currentText()
        output_function = self.output_function_options.currentText()
        epochs = int(self.epochs_spin.value())
        learning_rate = float(self.learning_rate_spin.value())

        self.main.start_training(self.layers, self.training_data, hidden_function, output_function, epochs,
                                 learning_rate)

    def open_test_data(self):
        self.test_data = None
        dlg = QFileDialog()
        dlg.setAcceptMode(QFileDialog.AcceptOpen)
        dlg.setFileMode(QFileDialog.ExistingFile)

        if dlg.exec_() != QDialog.Accepted:
            return

        file_name = dlg.selectedFiles()[0]
        if not file_name:
            return

        self.test_data = file_name
        print(self.test_data)

    def start_test(self):
        if self.training_data is None or self.test_data is None:
            return

        print(f'start to testing: {self.test_data}')
        self.main.start_testing(self.test_data)

    def add_hidden_layer(self):
        if len(self.layers) == 0:
            return

        self.layers.insert(len(self.layers) - 1, 1)
        self.print_layers()

    def print_layers(self):
        if len(self.layers) == 0:
            return

        self.remove_sub_layout()
        for index, layer_size in enumerate(self.layers):
            title = 'input' if index == 0 \
                else 'output' if index == len(self.layers) - 1 \
                else f'hidden {index}'

            layer_spin = QSpinBox()
            layer_spin.setObjectName(f'{index}')
            layer_spin.setRange(1, 2147483647)
            layer_spin.setValue(layer_size)
            layer_spin.valueChanged.connect(self.update_layer_value)

            layer_rmv_button = QPushButton('x')
            layer_rmv_button.setObjectName(f'{index}')
            layer_rmv_button.clicked.connect(self.remove_hidden_layer)

            g_layout = QGridLayout()
            g_layout.addWidget(QLabel(title), index, 0, 1, 1)
            g_layout.addWidget(layer_spin, index, 1, 1, 3)
            g_layout.addWidget(layer_rmv_button, index, 4, 1, 1)

            self.layout_right.addLayout(g_layout)

    def update_layer_value(self):
        index = int(self.sender().objectName())
        if index == 0 or index == len(self.layers) - 1:
            print('Cannot update input and output layer values')
            return

        if type(self.sender()) == QSpinBox:
            value = int(self.sender().value())
            if type(value) == int and type(index) == int:
                self.layers[index] = value

    def remove_hidden_layer(self):
        index = int(self.sender().objectName())
        if index == 0 or index == len(self.layers) - 1:
            print('Cannot delete input and output layer')
            return

        self.layers.pop(index)
        self.print_layers()

    def remove_sub_layout(self):
        while self.layout_right.count():
            item = self.layout_right.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                self.remove_sub_layout_item(item)

        # you could remove self.layout_right here

    def remove_sub_layout_item(self, item):
        if item.layout():
            while item.layout().count():
                child_item = item.layout().takeAt(0)
                widget = child_item.widget()
                if widget:
                    widget.deleteLater()
                else:
                    self.remove_sub_layout_item(child_item)

            item.layout().deleteLater()
        item.deleteLater()

    def set_center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
