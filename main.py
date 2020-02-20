# import necessary modules
import matplotlib.pyplot as plt
import numpy as np

# import kmeans custom class
from kmeans import Kmeans


# main
def main():
    # generate sampled points to plot as a line
    x = np.linspace(0, 1, 1000)
    y = 0.5 + 0.4 * np.sin(3 * np.pi * x)

    # generate uniform distribution features vector
    features = np.sort(np.random.uniform(0, 1, (75, 1)), axis=0)

    # generate uniform distribution noise vector
    noise = np.random.uniform(-0.1, 0.1, (75, 1))

    # generate the targets vector with added noise
    targets = 0.5 + 0.4 * np.sin(3 * np.pi * features) + noise

    kmeans = Kmeans(n_clusters=4)

    kmeans.fit(features)

    print(f'features = { features[0:10] }')
    centroids = kmeans.centroids
    print(f'centroids = { centroids[0:4] }')

    yf = np.linspace(0, 0, 75)
    plt.scatter(features, yf, color='red')

    yc = np.linspace(0, 0, 4)
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
