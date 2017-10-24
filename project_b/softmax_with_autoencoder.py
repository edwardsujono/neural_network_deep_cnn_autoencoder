import numpy as np
from common.utils import init_bias,init_weights
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class SoftmaxAutoEncoder:

    def __init__(self, num_features, num_outputs, list_hidden_layer, corruption_level,
                 learning_rate, sparsity_parameter, penalty_parameter):

        """
        :param list_hidden_layer: [10, 20] -> means 2 hidden layer, 10 neurons ->layer_1, 10 neurons -> layer_2
        :param corruption_level: made the noise in the data
        :param learning_rate: learning rate
        :param sparsity_parameter: 0.1 -> means 0.1 of neurons is activated
        :param penalty_parameter: learning decay
        """

        list_neurons = [num_features] + list_hidden_layer + [num_outputs]
        x = T.matrix('x')
        d = T.matrix('d')

        """
            Construct the auto encoder
        """

        weight_ae = self.init_weights(list_neurons[0], list_neurons[1], name_weight="weight_auto_encoder")
        bias_ae = self.init_bias(list_neurons[1], name_bias="bias_ae")
        bias_ae_pm = self.init_bias(list_neurons[0], name_bias="bias_ae_pm")

        y = T.nnet.sigmoid(T.dot(x, weight_ae) + bias_ae)
        z = T.nnet.sigmoid(T.dot(y, T.transpose(weight_ae)) + bias_ae_pm)

        cost = - T.mean(T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1))
        params = [weight_ae, bias_ae, bias_ae_pm]

        grads = T.grad(cost, params)

        updates = [(param - learning_rate*grad*param) for param, grad in zip(params, grads)]

        self.train_encoder = theano.function(
            inputs=[x],
            updates=updates,
            outputs=[z, cost]
        )

        """
            Construct the full connected neural network
        """

        prev_output = y
        list_weight = [weight_ae]
        list_bias = [bias_ae]

        for i in range(2, len(list_neurons)):

            weight = self.init_weights(list_neurons[i-1], list_neurons[i], 'weights_'+str(i))
            bias = self.init_bias(list_neurons[i], 'bias_'+str(i))

            if i < len(list_neurons)-1:
                prev_output = T.nnet.sigmoid(T.dot(prev_output, weight) + bias)
            else:
                # last layer need to do classification
                prev_output = T.nnet.softmax(T.dot(prev_output, weight) + bias)

            list_weight.append(weight)
            list_bias.append(bias)

        y_pred = T.argmax(prev_output, axis=1)

        cost_cross = T.mean(T.nnet.categorical_crossentropy(prev_output, d))
        params_cross = list_weight+list_bias
        grads_cross = T.grad(cost_cross, params_cross)
        updates_cross = [(param - learning_rate*grad*param) for param, grad in zip(params_cross, grads_cross)]

        self.train_cross = theano.function(
            inputs=[x, d],
            updates=updates_cross,
            outputs=[y_pred]
        )

    def start_train_auto_encoder(self, epochs, batch_size, train_x, train_y):

        print "Start training the auto encoder"
        d = []
        for epoch in range(epochs):
            # go through trainng set

            costs = []
            results = []

            for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_y), batch_size)):
                result, cost = self.train_encoder(train_x[start:end])
                costs.append(cost)
                results.append(result)

            d.append(np.mean(costs, dtype='float64'))
            print(d[epoch])
