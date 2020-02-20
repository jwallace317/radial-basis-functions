# import necessary modules
import matplotlib.pyplot as plt
import numpy as np

# import radial basis function neural network class
from rbf_network import RBFNeuralNetwork


# main
def main():
    # generate sampled points to plot as a line
    x = np.linspace(0, 1, 1000)
    y = 0.5 + 0.4 * np.sin(3 * np.pi * x)

    # generate uniform distribution features vector
    features = np.sort(np.random.uniform(0, 1, (9, 1)), axis=0)

    # generate uniform distribution noise vector
    noise = np.random.uniform(-0.1, 0.1, (9, 1))

    # generate the targets vector with added noise
    targets = 0.5 + 0.4 * np.sin(3 * np.pi * features) + noise

    # instantiate radial basis function neural network
    rbf_nn = RBFNeuralNetwork(n_clusters=3, max_epochs=100)

    # train the hidden layer of the rbf nn using kmeans algorithm
    rbf_nn.train_hidden_layer(features)

    print(f'features = { features[0:10] }')
    centroids = rbf_nn.centroids
    print(f'centroids = { centroids[0:3] }')

    rbf_nn.compute_gaussians(features[0])

    yf = np.linspace(0, 0, 9)
    plt.scatter(features, yf, color='red')

    yc = np.linspace(0, 0, 3)
    plt.scatter(centroids, yc, color='black')

    plt.show()

    # plot the figure
    # plt.figure(figsize=[25, 8])
    # # plt.scatter(centroids, centroids_y, color='black', label='centroids')
    # plt.scatter(features, targets, color='blue', label='sampled data')
    # plt.plot(x, y, color='red', label='sampling function')
    # plt.title('CSE 5526: Lab 2 Radial Basis Functions')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
