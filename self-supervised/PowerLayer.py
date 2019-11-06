import numpy as np

class Power:
    def __init__(self, dim):
        self.A = []
        self.dim = dim
        self.state = []

    def forward(self, inputs):
        self.A = inputs
        self.state = self.A
        A_real_square = np.power(self.A[:self.dim], 2)
        A_imag_square = np.power(self.A[self.dim:], 2)
        S = A_real_square + A_imag_square
        return S

    def backward(self, dydx):
        dxdz = 2 * np.hstack([np.diag(self.state[:self.dim]), np.diag(self.state[self.dim:])])
        power_grad = np.matmul(dydx, dxdz)
        power_grad = np.hstack([np.diag(power_grad[0,:self.dim].A1), np.diag(power_grad[0,self.dim:].A1)])
        return power_grad