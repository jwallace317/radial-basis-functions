# import necessary modules
import numpy as np


# kmeans alogrithm class
class Kmeans():

    # initialize kmeans
    def __init__(self, n_clusters, max_epochs, intra_cluster_variance):
        self.n_clusters = n_clusters  # number of clusters
        self.max_epochs = max_epochs  # max number of epochs for training
        self.centroids = np.zeros(n_clusters)  # centroid of each cluster
        self.variance = np.zeros(n_clusters)  # variance of each cluster
        self.intra_cluster_variance = intra_cluster_variance
        self.sse = 0  # sum of squared errors

    # initialize the starting centroids
    def initialize_centroids(self, features):
        centroids = np.random.permutation(features)[0:self.n_clusters]
        return centroids

    # compute distance from point to centroid
    def compute_distance(self, features, centroids):
        distance = np.zeros((features.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(features - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    # compute the centroids
    def compute_centroids(self, features, targets):
        centroids = np.zeros((self.n_clusters, features.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(features[targets == k, :], axis=0)
        return centroids

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

        # if variance = 0, reassign value to mean of the variances
        variance[variance == 0] = np.mean(variance)
        return variance

    def compute_constant_variance(self, centroids):
        max_distance = 0
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                distance = np.linalg.norm(centroids[i] - centroids[j])

                if distance > max_distance:
                    max_distance = distance

        variance = np.ones(self.n_clusters) * \
            np.square(max_distance / np.sqrt(2 * self.n_clusters))

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

        self.sse = self.compute_sse(features, clusters, self.centroids)

        if self.intra_cluster_variance is True:
            # compute intra cluster variance
            self.variance = self.compute_variance(features, clusters)
        else:
            # compute constant variance
            self.variance = self.compute_constant_variance(self.centroids)
