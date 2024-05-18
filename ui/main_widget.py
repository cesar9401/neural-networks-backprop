from PyQt5.QtWidgets import QWidget, QDesktopWidget, QGridLayout, QVBoxLayout, QGroupBox, QPushButton


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.functions = ['sigmoid', 'tanh', 'step', 'identity']
        self.setGeometry(100, 100, 800, 600)
        self.set_center()
        self.setWindowTitle("IA")

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
        layout_right = QVBoxLayout()
        layout_right.setSpacing(10)
        vertical_right_group.setLayout(layout_right)

        # open file
        self.open_training_btn = QPushButton("Open training data")
        self.open_training_btn.clicked.connect(self.open_training_data)
        layout.addWidget(self.open_training_btn)

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

        # add hidden layer test
        self.add_hidden_btn = QPushButton("Add hidden layer")
        self.add_hidden_btn.clicked.connect(self.add_hidden_layer)
        layout.addWidget(self.add_hidden_btn)

        self.grid.addLayout(self.left_layout, 0, 0)
        self.grid.addLayout(self.right_layout, 0, 1, 9, 9)

    def open_training_data(self):
        # TODO: open dialog here and select training data
        pass

    def start_training(self):
        # TODO: create network and start training
        pass

    def open_test_data(self):
        # TODO: open dialog here and select training data
        pass

    def start_test(self):
        # TODO: create network and start training
        pass

    def add_hidden_layer(self):
        # TODO: create network and start training
        pass

    def set_center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
