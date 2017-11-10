import theano
import theano.tensor as T
from common.utils import init_weight_biases_4dimension, init_weight_biases_2dimensions, shuffle_data
from theano.tensor.signal import pool
import numpy as np


class SoftmaxCNN:
    """ Softmax CNN Class """

    def __init__(self):
        self.conv_layers=[]
        self.pool_layers=[]
        self.hidden_layers=[]
        
        self.train = None
        self.predict = None
        self.Test = None
        
        self.learning = None
        self.learning_rate = None
        self.decay = None

        # For SGD learning
        self.momentum = None

        # For RMSProp
        self.rho = None
        self.epsilon = None


    def init_learning_sgd(self, learning_rate=0.05, decay=0.0001, momentum=0.0):
        """ Initialise parameter for SGD learning """
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        
        self.learning = self.sgd
        if momentum > 0.0:
            self.learning = self.sgd_momentum


    def init_learning_RMSProp(self, learning_rate=0.001, decay=0.0001, rho=0.9, epsilon=1e-6):
        """ Initialise parameter for RMSProp learning """
        self.learning_rate = learning_rate
        self.decay = decay
        self.rho = rho
        self.epsilon = epsilon

        self.learning = self.RMSprop


    def create_model(self, convolutional_layer, pooling_layer, hidden_layer, num_output):
        """
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

        for i in range(0, size_conv_pool, 1):
            weight_conv, bias_conv = init_weight_biases_4dimension(
                convolutional_layer[i], x.dtype)

            conv_out = T.nnet.relu(T.nnet.conv2d(prev_output, weight_conv) + bias_conv.dimshuffle('x', 0, 'x', 'x'))
            prev_output = pool.pool_2d(conv_out, pooling_layer[i])

            weights_conv.append(weight_conv)
            biases_conv.append(bias_conv)
            self.conv_layers.append(conv_out)
            self.pool_layers.append(prev_output)

        """
            Construct the full connected layer
        """
        prev_output = T.flatten(prev_output, outdim=2)

        weights = []
        biases = []

        for i in range(0, len(hidden_layer), 1):

            weight, bias = init_weight_biases_2dimensions(hidden_layer[i], x.dtype)

            prev_output = T.nnet.sigmoid(T.dot(prev_output, weight) + bias)

            weights.append(weight)
            biases.append(bias)
            self.hidden_layers.append(prev_output)

        weight, bias = init_weight_biases_2dimensions((hidden_layer[-1][1], num_output), x.dtype)

        weights.append(weight)
        biases.append(bias)

        prev_output = T.nnet.softmax(T.dot(prev_output, weight) + bias)

        prediction = T.argmax(prev_output, axis=1)

        cost = T.mean(T.nnet.categorical_crossentropy(prev_output, d))
        params = weights_conv + biases_conv + weights + biases

        updates = self.learning(cost, params)

        outputs = [cost] + self.conv_layers + self.pool_layers + self.hidden_layers
        
        self.train = theano.function(inputs=[x, d], outputs=outputs, updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[x], outputs=prediction, allow_input_downcast=True)
        self.test = theano.function(inputs=[x], outputs=outputs[1:], allow_input_downcast=True)


    def sgd(self, cost, params):
        """ Learning using SGD """
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            updates.append([p, p - (g + self.decay * p) * self.learning_rate])
        return updates


    def sgd_momentum(self, cost, params):
        """ Learning using SGD with momentum """
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            v = theano.shared(p.get_value())
            v_new = self.momentum*v - (g + self.decay*p) * self.learning_rate
            updates.append([p, p + v_new])
            updates.append([v, v_new])
            return updates


    def RMSprop(self, cost, params):
        """ Learning using RMSProp """
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = self.rho * acc + (1 - self.rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + self.epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - self.learning_rate * (g+ self.decay*p)))
        return updates


    def start_train(self, tr_x, tr_y, te_x, te_y, epochs, batch_size):

        self.predictions = []
        self.costs = []

        for i in range(epochs):

            tr_x, tr_y = shuffle_data(tr_x, tr_y)
            te_x, te_y = shuffle_data(te_x, te_y)

            costs = []

            for start, end in zip(range(0, len(tr_x), batch_size), range(batch_size, len(tr_y), batch_size)):
                outputs = self.train(tr_x[start:end], tr_y[start:end])
                costs.append(outputs[0])

            self.predictions.append(np.mean(np.argmax(te_y, axis=1) == self.predict(te_x)))
            self.costs.append(np.mean(costs))

            print('epoch: %d, accuracy: %s, cost: %s \n' % (i+1, self.predictions[i], self.costs[i]))
        
        return self.predictions, self.costs
