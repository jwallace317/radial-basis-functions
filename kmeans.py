# import necessary modules
import numpy as np


# kmeans alogrithm class
class Kmeans():

    # initialize kmeans
    def __init__(self, n_clusters, max_epochs):
        self.n_clusters = n_clusters  # number of clusters
        self.max_epochs = max_epochs  # max number of epochs for training
        self.centroids = np.zeros(n_clusters)  # centroid of each cluster
        self.variance = np.zeros(n_clusters)  # variance of each cluster
        self.error = 0  # sum of squared errors

    # initialize the starting centroids
    def initialize_centroids(self, features):
        centroids = np.random.permutation(features)[0:self.n_clusters]
        return centroids

    # compute the centroids
    def compute_centroids(self, features, targets):
        centroids = np.zeros((self.n_clusters, features.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(features[targets == k, :], axis=0)
        return centroids

    # compute distance from data point to centroid
    def compute_distance(self, features, centroids):
        distance = np.zeros((features.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(features - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    # find the closest centroid of a cluster to a data point
    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    # compute the sum of square errors
    def compute_sse(self, features, clusters, centroids):
        distance = np.zeros(features.shape[0])
        for k in range(self.n_clusters):
            distance[clusters == k] = np.linalg.norm(
                features[clusters == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    # compute the variance of each cluster
    def compute_variance(self, features, clusters):
        variance = np.zeros(self.n_clusters)
        for k in range(self.n_clusters):
            variance[k] = np.var(features[clusters == k])
        variance[variance == 0] = np.mean(variance)
        return variance

    # train the kmeans algorithm
    def train(self, features):
        self.centroids = self.initialize_centroids(features)
        for i in range(self.max_epochs):
            old_centroids = self.centroids
            distance = self.compute_distance(features, old_centroids)
            clusters = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(features, clusters)

            if np.all(old_centroids == self.centroids):
                break

        self.error = self.compute_sse(features, clusters, self.centroids)
        self.variance = self.compute_variance(features, clusters)
