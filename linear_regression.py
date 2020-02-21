# import necessary modules
import numpy as np
from sklearn.utils import shuffle


class LinearRegression():

    def __init__(self, variables):
        self.bias = 0
        self.weights = np.zeros((variables, 1))

    def fit(self, features, targets, learning_rate, max_epochs):
        for i in range(max_epochs):
            # randomly shuffle the data
            features, targets = shuffle(features, targets)

            for feature, target in zip(features, targets):
                linear_combination = np.dot(
                    feature.T, self.weights) + self.bias

                # backpropagation
                error = target - linear_combination

                self.bias += learning_rate * error
                self.weights += (learning_rate * error *
                                 feature).reshape(feature.shape[0], 1)
