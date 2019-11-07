import numpy as np

class Power:
    def __init__(self, num_beams):
        self.A = []
        self.num_beams = num_beams
        self.state = []

    def forward(self, inputs):
        self.A = inputs
        self.state = self.A
        A_real_square = np.power(self.A[:self.num_beams], 2)
        A_imag_square = np.power(self.A[self.num_beams:], 2)
        S = A_real_square + A_imag_square # S.shape: (num_beams,)
        return S

    def backward(self, dydx):
        dxdz = 2 * np.hstack([np.diag(self.state[:self.num_beams]), np.diag(self.state[self.num_beams:])])
        power_grad = np.matmul(dydx, dxdz)
        power_grad = np.hstack([np.diag(power_grad[0,:self.num_beams].A1), np.diag(power_grad[0, self.num_beams:].A1)])
        return power_grad