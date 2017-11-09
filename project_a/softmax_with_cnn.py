import theano
import theano.tensor as T
from common.utils import init_weight_biases_4dimension, init_weight_biases_2dimensions, shuffle_data
from theano.tensor.signal import pool
import numpy as np


class CNN:

    def __init__(self, learning_rate, decays,
                 convolutional_layer, pooling_layer, hidden_layer, num_output):

        """
        :param learning_rate: alpha 
        :param decays: beta
        :param convolutional_layer: list of convolutional [(15,1,9,9), (20,1,5,5)] -> first conv 15 kernels
                                    with size 9x9
        :param pooling_layer: list of ppoling layer [(2,2), (2,2)] -> first pool downsizing the cinv by 4x4
                                Notes that the the length of conv and pool layer need to be same
        :param hidden_layer: List of normal full connected layer [100]
        """
        theano.exception_verbosity='high'

        x = T.tensor4('x')
        d = T.matrix('d')

        weights_conv = []
        biases_conv = []

        # size conv and pool is strictly same
        size_conv_pool = len(convolutional_layer)

        prev_output = x

        """
            Construct the Convolutional together with the Pooling Layer
        """

        for i in xrange(0, size_conv_pool, 1):

            weight_conv, bias_conv = init_weight_biases_4dimension(
                convolutional_layer[i], x.dtype)

            y1 = T.nnet.relu(T.nnet.conv2d(prev_output, weight_conv) + bias_conv.dimshuffle('x', 0, 'x', 'x'))
            prev_output = pool.pool_2d(y1, pooling_layer[i])

            weights_conv.append(weight_conv)
            biases_conv.append(bias_conv)

        """
            Construct the full connected layer
        """
        prev_output = T.flatten(prev_output, outdim=2)

        weights = []
        biases = []

        for i in xrange(0, len(hidden_layer), 1):

            weight, bias = init_weight_biases_2dimensions(hidden_layer[i], x.dtype)

            prev_output = T.nnet.sigmoid(T.dot(prev_output, weight) + bias)

            weights.append(weight)
            biases.append(bias)

        weight, bias = init_weight_biases_2dimensions((hidden_layer[-1][1], num_output), x.dtype)

        weights.append(weight)
        biases.append(bias)

        prev_output = T.nnet.softmax(T.dot(prev_output, weight) + bias)

        prediction = T.argmax(prev_output, axis=1)

        cost = T.mean(T.nnet.categorical_crossentropy(prev_output, d))
        params = weights_conv + biases_conv + weights + biases

        updates = self.sgd(cost, params, lr=learning_rate, decay=decays)

        self.train = theano.function(inputs=[x, d], outputs=cost, updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[x], outputs=prediction, allow_input_downcast=True)
        # test = theano.function(inputs=[x], outputs=[y1, o1], allow_input_downcast=True)

    def sgd(self, cost, params, lr=0.05, decay=0.0001):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            updates.append([p, p - (g + decay * p) * lr])
        return updates

    def start_train(self, tr_x, tr_y, te_x, te_y, epochs, batch_size):

        predictions = []
        n_costs = []

        for i in range(epochs):

            tr_x, tr_y = shuffle_data(tr_x, tr_y)
            te_x, te_y = shuffle_data(te_x, te_y)

            costs = []

            for start, end in zip(range(0, len(tr_x), batch_size), range(batch_size, len(tr_y), batch_size)):

                cost = self.train(tr_x[start:end], tr_y[start:end])
                costs.append(cost)

            predictions.append(np.mean(np.argmax(te_y, axis=1) == self.predict(te_x)))
            n_costs.append(np.mean(costs))

            print 'accuracy: %s, cost: %s \n' % (predictions[i], n_costs[i])
