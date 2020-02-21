# import necessary modules
import numpy as np
from kmeans import Kmeans


class RBFNeuralNetwork():

    def __init__(self, n_clusters, max_epochs, learning_rate=0.01):
        # set class variables
        self.n_clusters = n_clusters
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate

        # initialize weights vector
        self.weights = np.zeros((n_clusters, 1))

        # initialize kmeans
        self.kmeans = Kmeans(n_clusters=n_clusters, max_epochs=max_epochs)

    def train_hidden_layer(self, features):
        self.kmeans.fit(features)
        self.centroids = self.kmeans.centroids

    def compute_gaussians(self, x):
        gaussians = np.zeros((self.n_clusters, 1))
        for k, (centroid, variance) in enumerate(zip(self.kmeans.centroids, self.kmeans.variance)):
            gaussians[k] = np.exp((-1 / (2 * variance))
                                  * (np.square(np.linalg.norm(x - centroid))))

        return gaussians

    def compute_sse(self, features, targets):
        sse = 0
        for feature, target in zip(features, targets):
            gaussians = self.compute_gaussians(feature)
            linear_combination = np.dot(gaussians.T, self.weights)
            error = linear_combination - target
            error = error**2
            sse += error

        return sse

    def predict(self, features):
        predict = []
        for feature in features:
            gaussians = self.compute_gaussians(feature)
            linear_combination = np.dot(gaussians.T, self.weights)
            predict.append(linear_combination)
        return predict

    def train_output_layer(self, features, targets):
        for i in range(self.max_epochs):
            for feature, target in zip(features, targets):
                gaussians = self.compute_gaussians(feature)

                # print(f'gaussians in train ouput = { gaussians }')

                linear_combination = np.dot(gaussians.T, self.weights)

                # print(f'linear combination = { linear_combination }')
                #
                # print(f'weights before update = { self.weights }')

                # backpropagation
                error = target - linear_combination

                self.weights += self.learning_rate * error * gaussians

                # print(f'weights after update = { self.weights }')

        sse = self.compute_sse(features, targets)
        print(f'sse after one epoch = { sse }')
        input()
