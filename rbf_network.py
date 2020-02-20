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
        gaussians = np.zeros(self.n_clusters)
        for k, (centroid, variance) in enumerate(zip(self.kmeans.centroids, self.kmeans.variance)):
            gaussians[k] = np.exp((-1 / (2 * variance))
                                  * (np.square(np.linalg.norm(x - centroid))))

        return gaussians


def train_output_layer(self, targets):

    return 0
