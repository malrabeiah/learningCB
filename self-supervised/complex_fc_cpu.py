import numpy as np

class FullyConnected:
    def __init__(self, num_beams, num_ant, mode='orig', accum=False): # mode: 'orig' (exact channel) or 'recon' (estimated channel)
        self.num_beams = num_beams
        self.num_ant = num_ant
        self.thetas = self.thetaInit()
        self.W = np.zeros([self.num_beams, self.num_ant])
        self.A = np.zeros([2*self.num_beams, 1])
        self.mode = mode
        self.state = []
        self.grad = np.zeros([self.num_beams, self.num_ant])
        self.accum = accum

    def thetaInit(self):
        init_thetas = 2 * np.pi * np.random.rand(self.num_beams, self.num_ant)
        return init_thetas

    def forward(self, ch): # ch: (2*num_ant, 1), organized as (h_r, h_i)
        if self.mode == 'orig':
            self.state = ch
        elif self.mode == 'recon':
            self.state = []
        else:
            ValueError('Set mode properly.')
        w_r = np.cos(self.thetas)
        w_i = np.sin(self.thetas)
        self.W = w_r + 1j * w_i
        W_block = np.block([[w_r, w_i],
                            [-w_i, w_r]])
        if ch.shape[0] == W_block.shape[1]:
            self.A = np.matmul(W_block, ch)
        else:
            ValueError('Error: dimensions mismatch, FullyConnected.forward!')
        return self.A

    def backward(self, dydx):
        h = np.zeros([1, 2*self.num_ant])
        if self.mode == 'orig':
            h = self.state
        elif self.mode == 'recon':
            h = self.estimate()
        else:
            ValueError('Set mode properly.')
        dxdz = np.zeros([2*self.num_beams, self.num_ant])
        for ii in range(self.num_beams):
            dxdz[ii,:] = -h[:self.num_ant]*np.sin(self.thetas[ii,:]) + h[self.num_ant:]*np.cos(self.thetas[ii,:])
            dxdz[ii+self.num_beams,:] = -h[self.num_ant:]*np.sin(self.thetas[ii,:]) - h[:self.num_ant]*np.cos(self.thetas[ii,:])
        if self.accum:
            self.grad += np.matmul(dydx, dxdz)
        else:
            self.grad = np.matmul(dydx, dxdz)
        return self.grad

    def estimate(self):
        A_complex = self.A[:self.num_beams] + 1j * self.A[self.num_beams:]
        W_conj = np.conj(self.W)
        h_est = np.matmul(np.linalg.pinv(W_conj, rcond=1e-3), A_complex)
        h_est = np.concatenate((np.real(h_est), np.imag(h_est)), axis=0)
        return h_est

    def update(self, lr=0.1):
        self.thetas = self.thetas - lr * self.grad
        return self.thetas

