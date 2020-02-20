# import necessary modules
import numpy as np
from numpy.linalg import norm


class Kmeans():

    # initialize kmeans
    def __init__(self, n_clusters=2, max_epochs=100):
        self.n_clusters = n_clusters
        self.max_epochs = max_epochs

    # initialize the starting centroids
    def initialize_centroids(self, X):
        centroids = np.random.permutation(X)[0:self.n_clusters]
        return centroids

    # compute the centroids
    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    # compute distance from data point to centroid
    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)

        return distance

    # find the closest centroid of a cluster to a data point
    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    # compute the sum of square errors
    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    # fit the provided data
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for i in range(self.max_epochs):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)

            if np.all(old_centroids == self.centroids):
                break

        self.error = self.compute_sse(X, self.labels, self.centroids)
