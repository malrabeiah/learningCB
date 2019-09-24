'''
An implementation of complex fully-connected layer for codebook learning.
It is mainly a modification on the implementation of Chiheb Trabelsi, found in:
https://github.com/ChihebTrabelsi/deep_complex_networks

Author: Muhammad Alrabeiah
Sept. 2019
'''
import sys;sys.path.append('.')
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.layers import Layer, InputSpec
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class CompFC(Layer):

    def __init__(self, units,
                 activation=None,
                 init_criterion='he',
                 kernel_initializer='complex',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 seed=None,
                 scale=1,
                 **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(CompFC, self).__init__(**kwargs)
        self.units = units
        self.scale = scale
        self.activation = activations.get(activation)
        self.init_criterion = init_criterion
        if kernel_initializer in {'complex', 'constant'}:
            self.kernel_initializer = kernel_initializer
        else:
            self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 10e6)
        else:
            self.seed = seed
        self.input_spec = InputSpec(ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[-1] % 2 == 0
        input_dim = input_shape[-1] // 2
        data_format = K.image_data_format()
        kernel_shape = (input_dim, self.units)
        fan_in, fan_out = initializers._compute_fans(
            kernel_shape,
            data_format=data_format
        )

        if self.init_criterion == 'he':
            s = K.sqrt(1. / fan_in)
        elif self.init_criterion == 'glorot':
            s = K.sqrt(1. / (fan_in + fan_out))
        rng = RandomStreams(seed=self.seed)


        def init_theta(shape, dtype=None):
            return rng.uniform(size=kernel_shape, low=0, high=6)

        if self.kernel_initializer in {'complex'}:
            theta_init = init_theta
        else:
            raise ValueError('Not recognized choice of initialization!')

        # Defining layer parameters (Codebook):
        self.theta = self.add_weight(
            shape=kernel_shape,
            initializer=theta_init,
            name='theta_kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )

        self.real_kernel = (1 / self.scale) * K.cos(self.theta)  #
        self.imag_kernel = (1 / self.scale) * K.sin(self.theta)  #

        self.input_spec = InputSpec(ndim=2, axes={-1: 2 * input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.shape(inputs)
        input_dim = input_shape[-1] // 2
        real_input = inputs[:, :input_dim]
        imag_input = inputs[:, input_dim:]

        cat_kernels_4_real = K.concatenate(
            [self.real_kernel, -self.imag_kernel],
            axis=-1
        )
        cat_kernels_4_imag = K.concatenate(
            [self.imag_kernel, self.real_kernel],
            axis=-1
        )
        cat_kernels_4_complex = K.concatenate(
            [cat_kernels_4_real, cat_kernels_4_imag],
            axis=0
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]

        output = K.dot(inputs, cat_kernels_4_complex)

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 2 * self.units
        return tuple(output_shape)

    def get_config(self):
        if self.kernel_initializer in {'complex'}:
            ki = self.kernel_initializer
        else:
            ki = initializers.serialize(self.kernel_initializer)
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'init_criterion': self.init_criterion,
            'kernel_initializer': ki,
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'seed': self.seed,
        }
        base_config = super(CompFC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
