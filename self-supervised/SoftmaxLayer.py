import numpy as np

class Softmax:
    def __init__(self, dim):
        self.S = []
        self.dim = dim
        self.den = 0

    def forward(self, inputs):
        self.S = inputs - np.max(inputs)
        self.den = sum(np.exp(self.S))
        P = np.exp(self.S)/self.den
        return P

    def backward(self, dydx):
        exp_vec = np.mat(np.exp(self.S).reshape([self.dim, 1]))
        exp_mat = (-1/np.power(self.den, 2)) * np.matmul(exp_vec, np.transpose(exp_vec))
        dxdz = (1/self.den) * np.diag(exp_vec) + exp_mat
        softmax_grad = np.matmul(dydx, dxdz)
        return softmax_grad