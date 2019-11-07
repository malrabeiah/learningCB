import numpy as np
import sys

# for one-hot vector label only
class CrossEntroy:
    def __init__(self):
        self.prob = np.array([])
        self.label = np.array([])
        self.loss = 0

    def forward(self, prob, label):
        self.prob = prob.reshape((1, prob.size))
        self.label = label.reshape((1, label.size))
        pos_label = self.label.argmax()
        pos_prob = self.prob.argmax()
        if pos_label != pos_prob:
            print('Inconsistent maximal position!')
            sys.exit()
        self.loss = -np.log(self.prob[0, pos_prob]) # np.log is natural logarithm
        return self.loss

    def backward(self):
        pos = self.label.argmax()
        grad = -1/self.prob[0, pos]
        dydx = np.zeros([1, self.label.size])
        dydx[0, pos] = grad
        return dydx # dydx.shape: (1, num_beams)