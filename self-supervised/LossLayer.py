import numpy as np

class CrossEntroy:
    def __init__(self):
        self.P = np.array([])
        self.label = np.array([])
        self.loss = 0

    def forward(self, prob, label):
        self.P = prob.reshape((1, prob.size))
        self.label = label.reshape((1, label.size))
        self.loss = -np.sum(self.label * np.log(self.P))
        return self.loss

    def backward(self):
        dydx = -(self.label/self.P)
        return dydx