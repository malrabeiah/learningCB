#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Authors: Chiheb Trabelsi
#

import sys;sys.path.append('.')
from keras import backend as K
from keras.layers import Layer


class NewLayer_AA(Layer):

    def __init__(self, in_shape,batch_size):
        super(NewLayer_AA, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)
        self.batch_size = batch_size

    def call(self, x):
        real_part = x[:,:self.len_real]
        imag_part = x[:,self.len_real:]
        sq_real =  K.pow(real_part,2)
        sq_imag =  K.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        max_pooling = K.reshape(abs_values,(self.batch_size,1,self.len_real))
        return max_pooling

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        S = list(input_shape)
        output_shape = ( S[0],1,int(S[1]/2) )
        return tuple(output_shape)

    # def get_config(self):
    #     if self.kernel_initializer in {'complex'}:
    #         ki = self.kernel_initializer
    #     else:
    #         ki = initializers.serialize(self.kernel_initializer)
    #     config = {
    #         'units': self.units,
    #         'activation': activations.serialize(self.activation),
    #         'use_bias': self.use_bias,
    #         'init_criterion': self.init_criterion,
    #         'kernel_initializer': ki,
    #         'bias_initializer': initializers.serialize(self.bias_initializer),
    #         'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
    #         'bias_regularizer': regularizers.serialize(self.bias_regularizer),
    #         'activity_regularizer': regularizers.serialize(self.activity_regularizer),
    #         'kernel_constraint': constraints.serialize(self.kernel_constraint),
    #         'bias_constraint': constraints.serialize(self.bias_constraint),
    #         'seed': self.seed,
    #     }
    #     base_config = super(ComplexDense, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

