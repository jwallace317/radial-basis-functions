# import necessary modules
import numpy as np
from sklearn.utils import shuffle

from kmeans import Kmeans
from linear_regression import LinearRegression


# radial basis function neural network class
class RBFNeuralNetwork():

    # initialize radial basis function neural network
    def __init__(self, n_clusters, max_epochs, learning_rate):
        self.n_clusters = n_clusters  # number of clusters in kmeans algorithm
        self.max_epochs = max_epochs  # max number of epochs allowed for training
        self.learning_rate = learning_rate  # learning rate of regression
        self.sse = 0  # sum of squared errors

        # instantiate kmeans class to perform kmeans clustering
        self.kmeans = Kmeans(n_clusters=n_clusters, max_epochs=max_epochs)
        self.linear_regression = LinearRegression(n_clusters)

    # compute the gaussian values at each hidden node
    def compute_gaussians(self, features):
        gaussians = np.zeros((features.shape[0], self.n_clusters))
        for i, feature in enumerate(features):
            for k, (centroid, variance) in enumerate(zip(self.kmeans.centroids, self.kmeans.variance)):
                gaussians[i, k] = np.exp(
                    (-1 / (2 * variance)) * (np.square(np.linalg.norm(feature - centroid))))
        return gaussians

    # compute the sum of squared errors
    def compute_sse(self, features, targets):
        gaussians = self.compute_gaussians(features)
        outputs = np.dot(gaussians, self.linear_regression.weights) + \
            self.linear_regression.bias
        errors = np.square(targets - outputs)
        return np.sum(errors)

    # predict targets using radial basis function neural network given features
    def predict(self, features):
        gaussians = self.compute_gaussians(features)
        predict = np.dot(gaussians, self.linear_regression.weights) + \
            self.linear_regression.bias
        return predict

    # train the hidden layer by means of kmeans clustering
    def train_hidden_layer(self, features):
        self.kmeans.train(features)

    # train the output layer by means of LMS multivariate linear regression
    def train_output_layer(self, features, targets):
        gaussians = self.compute_gaussians(features)
        self.linear_regression.fit(
            gaussians, targets, self.learning_rate, self.max_epochs)

    # train the hidden layer and output layer with features and targets
    def train(self, features, targets):
        self.train_hidden_layer(features)
        self.train_output_layer(features, targets)
