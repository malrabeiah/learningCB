import numpy as np
from complex_fc_cpu import FullyConnected
from PowerLayer import Power
from SoftmaxLayer import Softmax
from ArgmaxLayer import Argmax
from LossLayer import CrossEntroy

class Model:
    def __init__(self, num_beams, num_ant, mode='orig', accum=False):
        # layers
        self.ComplexFC = FullyConnected(num_beams, num_ant, mode, accum)
        self.Power = Power(num_beams)
        self.SoftMax = Softmax(num_beams)
        self.ArgMax = Argmax(num_beams)
        self.Loss = CrossEntroy()
        # codebook and gradient
        self.codebook = self.ComplexFC.thetas
        self.grad = self.ComplexFC.grad

    def forward(self, h):
        A = self.ComplexFC.forward(h)   # A.shape: (2*num_beams,) organized as (a_r, a_i)
        S = self.Power.forward(A)       # S.shape: (num_beams,)
        P = self.SoftMax.forward(S)     # P.shape: (num_beams,)
        L = self.ArgMax.forward(S)      # L.shape: (1, num_beams)
        loss = self.Loss.forward(P, L)
        return loss

    def backward(self):
        dL_dP = self.Loss.backward() # dL_dP.shape: (1, num_beams)
        dL_dS = self.SoftMax.backward(dL_dP)
        dL_dA = self.Power.backward(dL_dS)
        dydx = self.ComplexFC.backward(dL_dA)
        self.grad = dydx
        return dydx

    def update(self, lr=0.1):
        self.codebook = self.ComplexFC.update(lr=lr)
        return self.codebook