import numpy as np
from common.utils import init_bias, init_weights, shuffle_data, init_sparsity_constraint, sgd_momentum
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class SoftmaxAutoEncoder:
    def __init__(self, num_features, num_outputs, list_hidden_layer, learning_rate,
                 corruption_level=0.01,
                 sparsity_parameter=0.01, penalty_parameter=0.01, momentum=0.1,
                 use_sparsity=False, use_momentum=False, xavier_range=4, shared_weight=True):

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
        self.list_prev_output = []
        self.reconstructed_image = None
        self.hidden_layer_value = []
        self.train_encoder_function = []

        """
            Do the data corruption
        """

        tilde_x = self.corrupt_the_data(corruption_level, x)

        """
            Construct the auto encoder
        """

        self.weights = []
        self.weights_transpose = []
        biases = []
        biases_trans = []

        check_input_train = None
        check_output_train = None

        for ind_neurons in range(2, len(list_neurons) + 1):

            """
                ENCODER
            """
            list_trained_neurons = list_neurons[:ind_neurons]
            prev_output = tilde_x
            self.list_prev_output = []

            for i in range(1, len(list_trained_neurons)):

                if i == ind_neurons - 1:
                    weight = init_weights(list_trained_neurons[i - 1], list_trained_neurons[i],
                                          'weight_%s' % i, xavier_range)
                    bias = init_bias(list_trained_neurons[i], 'bias_%s' % i)

                    self.weights.append(weight)
                    biases.append(bias)
                    check_input_train = prev_output
                    if i == 1:
                        check_input_train = x

                else:
                    # use the previous value, we solely trained the added layer
                    weight = self.weights[i - 1]
                    bias = biases[i - 1]

                prev_output = T.nnet.sigmoid(T.dot(prev_output, weight) + bias)
                self.list_prev_output.append(prev_output)

            """
                DECODER
            """

            buffer_output = prev_output

            for i in range(len(self.weights) - 1, -1, -1):

                if i == ind_neurons - 2:
                    bias_transpose = init_bias(list_trained_neurons[i], 'bias_trans_%s' % i)
                    biases_trans.append(bias_transpose)
                else:
                    # use the previous bias transpose
                    bias_transpose = biases_trans[i]

                weight_transpose = self.weights[i].transpose()

                # instanstiate new weight if not shared weight
                if not shared_weight:

                    if i == ind_neurons - 2:
                        eval_size = weight_transpose.eval().shape
                        weight_transpose = init_weights(eval_size[0], eval_size[1], 'weight_transpose_%s' % i,
                                                        xavier_range)
                        self.weights_transpose.append(weight_transpose)
                    else:
                        weight_transpose = self.weights_transpose[i]

                prev_output = T.nnet.sigmoid(T.dot(prev_output, weight_transpose) + bias_transpose)

                if i == ind_neurons - 2:
                    check_output_train = prev_output

            cost = - T.mean(T.sum(check_input_train * T.log(check_output_train) +
                                  (1 - check_input_train) * T.log(1 - check_output_train), axis=1))

            if use_sparsity:
                cost += init_sparsity_constraint(list_neurons=self.list_prev_output,
                                                 sparsity_parameter=sparsity_parameter,
                                                 penalty_parameter=penalty_parameter)

            params = [self.weights[ind_neurons - 2]] + [biases[ind_neurons - 2]] + [biases_trans[ind_neurons - 2]]

            if not shared_weight:
                params += [self.weights_transpose[ind_neurons - 2]]

            if not use_momentum:
                grads = T.grad(cost, params)
                updates = [(param, param - learning_rate * grad) for param, grad in zip(params, grads)]
            else:
                updates = sgd_momentum(cost, params, momentum=momentum)

            outputs = [cost, prev_output]
            outputs.extend([p_output for p_output in self.list_prev_output])

            train_encoder = theano.function(
                inputs=[x],
                updates=updates,
                outputs=outputs
            )
            self.train_encoder_function.append(train_encoder)

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
        self.total_costs_auto_encoder = []

        cnt_layer = 1

        for train_encoder in self.train_encoder_function:

            print "Start training layer %s " % cnt_layer
            current_costs_auto_encoder = []
            train_x, train_y = shuffle_data(train_x, train_y)

            for epoch in range(epochs):
                # go through training set

                costs = []

                for start, end in zip(range(0, len(train_x), batch_size), range(batch_size, len(train_y), batch_size)):

                    self.hidden_layer_value = []

                    all_value = train_encoder(train_x[start:end])
                    cost, self.reconstructed_image = all_value[:2]

                    for val in all_value[2:]:
                        self.hidden_layer_value.append(val)

                    costs.append(cost)

                current_costs_auto_encoder.append(np.mean(costs, dtype='float64'))

                if verbose:
                    print "Epoch: %d Cost: %s \n" % (epoch, current_costs_auto_encoder[epoch])

            self.total_costs_auto_encoder.append(current_costs_auto_encoder)

            print "Finish training layer %s " % cnt_layer
            cnt_layer += 1

    def start_train_the_full(self, epochs, batch_size, train_x, train_y, test_x, test_y):

        print "Start training the full hidden layer with autoencoder"
        self.total_costs_full = []
        self.total_predictions_full = []

        for epoch in range(epochs):
            # go through trainng set
            costs = []
            results = []
            train_x, train_y = shuffle_data(train_x, train_y)

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

    def get_weights_on_each_layer(self):

        return [weight.get_value() for weight in self.weights]

    def get_reconstructed_image(self):

        return self.reconstructed_image

    def get_hidden_layer_activation(self):

        return self.hidden_layer_value
