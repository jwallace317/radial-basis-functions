# import necessary modules
import numpy as np
from kmeans import Kmeans
from sklearn.utils import shuffle


# radial basis function neural network class
class RBFNeuralNetwork():

    # initialize radial basis function neural network
    def __init__(self, n_clusters, max_epochs, learning_rate):
        self.n_clusters = n_clusters  # number of clusters in kmeans algorithm
        self.max_epochs = max_epochs  # max number of epochs allowed for training
        self.learning_rate = learning_rate  # learning rate of regression
        self.bias = 0  # bias of output layer
        self.weights = np.zeros((n_clusters, 1))  # weights of output layer
        self.sse = 0  # sum of squared errors

        # instantiate kmeans class to perform kmeans clustering
        self.kmeans = Kmeans(n_clusters=n_clusters, max_epochs=max_epochs)

    # train the hidden layer by means of kmeans clustering
    def train_hidden_layer(self, features):
        self.kmeans.train(features)

    # compute the gaussian values at each hidden node
    def compute_gaussians(self, feature):
        gaussians = np.zeros((self.n_clusters, 1))
        for k, (centroid, variance) in enumerate(zip(self.kmeans.centroids, self.kmeans.variance)):
            gaussians[k] = np.exp((-1 / (2 * variance))
                                  * (np.square(np.linalg.norm(feature - centroid))))
        return gaussians

    # compute the sum of squared errors
    def compute_sse(self, features, targets):
        sse = 0
        for feature, target in zip(features, targets):
            gaussians = self.compute_gaussians(feature)
            linear_combination = np.dot(gaussians.T, self.weights) + self.bias
            error = linear_combination - target
            error = error**2
            sse += error
        return sse

    # predict targets using radial basis function neural network given features
    def predict(self, features):
        predict = []
        for feature in features:
            gaussians = self.compute_gaussians(feature)
            linear_combination = np.dot(gaussians.T, self.weights) + self.bias
            predict.append(linear_combination)
        return predict

    # train the output layer by means of LMS multivariate linear regression
    def train_output_layer(self, features, targets):
        for i in range(self.max_epochs):
            # randomly shuffle the data
            features, targets = shuffle(features, targets)

            for feature, target in zip(features, targets):
                gaussians = self.compute_gaussians(feature)
                linear_combination = np.dot(
                    gaussians.T, self.weights) + self.bias

                # backpropagation
                error = target - linear_combination

                self.bias += self.learning_rate * error
                self.weights += self.learning_rate * error * gaussians

        self.sse = self.compute_sse(features, targets)
