class Data:
    def __init__(self, train, train_labels, test, test_labels):
        self.train = train / 255
        self.train_labels = train_labels
        self.test = test / 255
        self.test_labels = test_labels
