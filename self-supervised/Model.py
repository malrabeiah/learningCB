import numpy as np
from complex_fc_cpu import FullyConnected
from PowerLayer import Power
from SoftmaxLayer import Softmax
from ArgmaxLayer import Argmax
from LossLayer import CrossEntroy

class Model:
    def __init__(self, num_beams, num_ant, mode='orig', accum='False'):
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
        A = self.ComplexFC.forward(h)
        S = self.Power.forward(A)
        P = self.SoftMax.forward(S)
        I = self.ArgMax.forward(S)
        loss = self.Loss.forward(P, I)
        print('Loss: %f' % loss)

    def backward(self):
        dydx = self.Loss.backward()
        dydx = self.SoftMax.backward(dydx)
        dydx = self.Power.backward(dydx)
        dydx = self.ComplexFC.backward(dydx)
        self.grad = dydx
        return dydx

    def update(self, lr=0.01):
        self.codebook = self.codebook - lr * self.grad
        return self.codebook