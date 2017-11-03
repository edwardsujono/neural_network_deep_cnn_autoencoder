import numpy as np
import theano
import theano.tensor as T


def init_weights(n_in, n_out, name_weight):
    weight = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_in + n_out)),
            high=4 * np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)),
        dtype=theano.config.floatX)
    return theano.shared(value=weight, name=name_weight, borrow=True)


def init_bias(n, name_bias):
    return theano.shared(value=np.zeros(n, dtype=theano.config.floatX), name=name_bias, borrow=True)


def init_sparsity_constraint(list_back_neurons, penalty_parameter, sparsity_parameter):


    computation = 0

    for back_neuron in list_back_neurons:

        computation += penalty_parameter * T.shape(back_neuron)[1] * (sparsity_parameter *
        T.log(sparsity_parameter) +
        (1 - sparsity_parameter) * T.log(1 - sparsity_parameter)) \
        - penalty_parameter * sparsity_parameter * T.sum(T.log(T.mean(back_neuron, axis=0) + 1e-6)) \
        - penalty_parameter * (1 - sparsity_parameter) * T.sum(T.log(1 - T.mean(back_neuron, axis=0) + 1e-6))

    return computation


def init_weight_biases_4dimension(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values = np.asarray(
        np.random.uniform(low=-bound, high=bound, size=filter_shape),
        dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values, borrow=True), theano.shared(b_values, borrow=True)


def init_weight_biases_2dimensions(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values = np.asarray(
        np.random.uniform(low=-bound, high=bound, size=filter_shape),
        dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values, borrow=True), theano.shared(b_values, borrow=True)


def shuffle_data(samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


def sgd_momentum(cost, params, lr=0.05, decay=0.0001, momentum=0.5):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value())
        v_new = momentum*v - (g + decay*p) * lr
        updates.append([p, p + v_new])
        updates.append([v, v_new])
        return updates
