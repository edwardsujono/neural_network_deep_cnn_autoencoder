import numpy as np
import theano


def init_weights(self, n_in, n_out, name_weight):
    weight = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_in + n_out)),
            high=4 * np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)),
        dtype=theano.config.floatX)
    return theano.shared(value=weight, name=name_weight, borrow=True)


def init_bias(self, n, name_bias):
    return theano.shared(value=np.zeros(n, dtype=theano.config.floatX), name=name_bias, borrow=True)