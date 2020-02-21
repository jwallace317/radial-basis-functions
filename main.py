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

    # generate uniform distribution in [0, 1) features vector
    features = np.sort(np.random.uniform(0, 1, (75, 1)), axis=0)

    # generate uniform distribution in [-0.1, 0.1) noise vector
    noise = np.random.uniform(-0.1, 0.1, (75, 1))

    # generate the targets vector with added noise
    targets = 0.5 + 0.4 * np.sin(3 * np.pi * features) + noise

    n_clusters = [3, 6, 8, 12, 16]
    learning_rates = [0.01, 0.02]

    for learning_rate in learning_rates:
        for k in n_clusters:

            # instantiate radial basis function neural network
            rbf_nn = RBFNeuralNetwork(
                n_clusters=k, max_epochs=100, learning_rate=learning_rate)

            # train the hidden layer
            rbf_nn.train_hidden_layer(features)

            # train the output layer
            rbf_nn.train_output_layer(features, targets)

            # predict the targets given the features
            predicted = rbf_nn.predict(features)

            # compute the sum of squared errors
            sse = rbf_nn.compute_sse(features, targets).item()

            # plot results
            plt.figure(figsize=[12, 8])
            plt.plot(x, y, color='black', label='original sampling function')
            plt.scatter(features, targets, color='red', label='targets')
            plt.scatter(features, predicted, color='blue',
                        label='predicted targets')
            plt.title(
                'Radial Basis Function Neural Network Function Approximation')
            plt.text(0.8, 0.1, f'number of clusters = { k }')
            plt.text(0.8, 0.06, f'learning rate = { learning_rate }')
            plt.text(0.8, 0.03, f'sse = { sse }')
            plt.legend()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

            input()


if __name__ == '__main__':
    main()
