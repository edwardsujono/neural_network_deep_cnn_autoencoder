""" 
Data collector for mnist.pkl 
"""

import pickle
import numpy as np


class DataCollector:
    """ DataCollector class """

    def __init__(self, file_path):
        with open(file_path, "rb") as input_file:
            self.data = pickle.load(input_file, encoding="latin-1")

        self.data_train = self.data[0]
        self.data_test = self.data[2]
        self.data_validate = self.data[1]

    def get_train_data(self):
        """ get train data with one hot encoding """
        return self.data_train[0], self.return_one_hot_encoding(10, self.data_train[1])

    def get_test_data(self):
        """ get test data with one hot encoding """
        return self.data_test[0], self.return_one_hot_encoding(10, self.data_test[1])

    def get_validation_data(self):
        """ get validation data with one hot encoding """
        return self.data_validate[0], self.return_one_hot_encoding(10, self.data_validate[1])

    def return_one_hot_encoding(self, num_output, list_data):
        """ One hot encoding """
        zeros = np.zeros((len(list_data), num_output))

        for i in range(len(zeros)):
            zeros[i][list_data[i]] = 1

        return zeros
