import numpy as np
from common.utils import init_bias, init_weights
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class SoftmaxAutoEncoder:
    def __init__(self, num_features, num_outputs, list_hidden_layer, learning_rate,
                 corruption_level=0.01,
                 sparsity_parameter=0.01, penalty_parameter=0.01):

        """
        :param list_hidden_layer: [10, 20] -> means 2 hidden layer, 10 neurons ->layer_1, 10 neurons -> layer_2
        :param corruption_level: made the noise in the data
        :param learning_rate: learning rate
        :param sparsity_parameter: 0.1 -> means 0.1 of neurons is activated
        :param penalty_parameter: learning decay
        """

        list_neurons = [num_features] + list_hidden_layer
        x = T.matrix('x')
        d = T.matrix('d')

        self.total_costs_auto_encoder = []
        self.total_costs_full = []
        self.total_predictions_full = []

        """
            Do the data corruption
        """

        x = self.corrupt_the_data(corruption_level=corruption_level, x=x)

        """
            Construct the auto encoder
        """

        self.weights = []
        biases = []

        prev_output = x

        """
            ENCODER
        """

        for i in range(1, len(list_neurons)):

            weight = init_weights(list_neurons[i-1], list_neurons[i], 'weight_%s' % i)
            bias = init_bias(list_neurons[i], 'bias_%s' % i)

            self.weights.append(weight)
            biases.append(bias)

            prev_output = T.nnet.sigmoid(T.dot(prev_output, weight) + bias)

        """
            DECODER
        """

        buffer_output = prev_output
        biases_trans = []

        for i in range(len(self.weights)-1, -1, -1):

            weight_transpose = self.weights[i].transpose()
            bias_transpose = init_bias(list_neurons[i], 'bias_trans_%s' % i)

            biases_trans.append(bias_transpose)

            prev_output = T.nnet.sigmoid(T.dot(prev_output, weight_transpose) + bias_transpose)

        # last_output = T.switch(T.gt(prev_output, 0.5), 1, 0)
        cost = - T.mean(T.sum(x * T.log(prev_output) + (1 - x) * T.log(1 - prev_output), axis=1))
        # cost = T.mean(T.nnet.binary_crossentropy(prev_output, x))

        params = self.weights+biases+biases_trans
        grads = T.grad(cost, params)
        updates = [(param, param - learning_rate*grad) for param, grad in zip(params, grads)]
        outputs = [cost]

        self.train_encoder = theano.function(
            inputs=[x],
            updates=updates,
            outputs=outputs
        )

        """
            TRAIN THE FULL CONNECTED LAYER
        """

        last_weight = init_weights(list_neurons[-1], num_outputs, 'last_weight')
        last_bias = init_bias(num_outputs, 'last_bias')

        buffer_output = T.nnet.softmax(T.dot(buffer_output, last_weight) + last_bias)
        y_pred = T.argmax(buffer_output, axis=1)

        cost_cross = T.mean(T.nnet.categorical_crossentropy(buffer_output, d))

        params_full = self.weights + [last_weight] + biases + [last_bias]
        grads_full = T.grad(cost_cross, params_full)
        updates_full = [(param, param - learning_rate * grad) for param, grad in zip(params_full, grads_full)]

        self.train_cross = theano.function(
            inputs=[x, d],
            updates=updates_full,
            outputs=[prev_output, y_pred, cost_cross]
        )

        self.test_cross = theano.function(
            inputs=[x],
            outputs=[y_pred]
        )

    def corrupt_the_data(self, corruption_level, x):

        # use binomial dist at corrupt the data
        rng = np.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                                      dtype=theano.config.floatX) * x

        return tilde_x

    def start_train_auto_encoder(self, epochs, batch_size, train_x, train_y, verbose=False):

        print "Start training the auto encoder"

        for epoch in range(epochs):
            # go through trainig set

            costs = []

            for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_y), batch_size)):
                cost = self.train_encoder(train_x[start:end])
                costs.append(cost)

            self.total_costs_auto_encoder.append(np.mean(costs, dtype='float64'))

            if verbose:
                print "Epoch: %d Cost: %s \n" % (epoch, self.total_costs_auto_encoder[epoch])

    def start_train_the_full(self, epochs, batch_size, train_x, train_y, test_x, test_y):

        print "Start training the full hidden layer with autoencoder"

        for epoch in range(epochs):
            # go through trainng set
            costs = []
            results = []

            for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_y), batch_size)):
                output, result, cost = self.train_cross(train_x[start:end], train_y[start:end])
                costs.append(cost)
                results.append(np.mean(np.argmax(test_y, axis=1) == self.test_cross(test_x)))

            self.total_costs_full.append(np.mean(costs, dtype='float64'))
            self.total_predictions_full.append(np.mean(results, dtype='float64'))
            print "cost: %s, prediction: %s \n" % (self.total_costs_full[epoch], self.total_predictions_full[epoch])

    def get_total_costs_of_auto_encoder(self):

        return self.total_costs_auto_encoder

    def get_total_cost_and_prediction_full(self):

        return self.total_costs_full, self.total_predictions_full

    def get_output_on_each_layer(self):

        return [weight.get_value() for weight in self.weights]
