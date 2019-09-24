'''
An implementation of Absolute units and max-pooling.
Author: Muhammad Alrabeiah
Sept. 2019
'''
import sys;sys.path.append('.')
from keras import backend as K
from keras.layers import Layer


class PowerPooling(Layer):

    def __init__(self, in_shape):
        super(PowerPooling, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)

    def call(self, x):
        real_part = x[:,:self.len_real]
        imag_part = x[:,self.len_real:]
        sq_real = K.pow(real_part,2)
        sq_imag = K.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        max_pooling = K.max(abs_values, axis=-1)
        max_pooling = K.expand_dims(max_pooling,axis=-1)
        return max_pooling

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        S = list(input_shape)
        output_shape = S[0]
        output_shape = [output_shape,1]
        return tuple(output_shape)


