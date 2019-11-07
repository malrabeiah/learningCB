import numpy as np

class Argmax:
    def __init__(self, dim):
        self.S = []
        self.dim = dim

    def forward(self, inputs):
        self.S = inputs
        max_pos = self.S.argmax()
        one_hot = np.zeros([1, self.dim])
        one_hot[0, max_pos] = 1 # one_hot.shape: (1, num_beams)
        return one_hot