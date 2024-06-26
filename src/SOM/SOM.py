import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class SOM:
    def __init__(self, m, n, dim, n_iterations=100, alpha_start=0.6, sigma_start=None):
        self.m = m
        self.n = n
        self.dim = dim
        self.n_iterations = n_iterations
        self.alpha_start = alpha_start
        self.sigma_start = sigma_start if sigma_start else max(m, n) / 2.0
        self.locations = np.array(list(self._neuron_locations(m, n)))

    def _initialize_weights_from_data(self, data):
        indices = np.random.choice(range(data.shape[0]), size=(self.m, self.n))
        return data[indices]

    def _neuron_locations(self, m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def _neighborhood(self, center, sigma):
        d = 2 * np.pi * sigma ** 2
        ax = np.exp(-np.power(np.linalg.norm(self.locations - center, axis=1), 2) / d)
        return ax.reshape((self.m, self.n))

    def _learning_rate(self, iteration, n_iterations):
        return self.alpha_start * np.exp(-iteration / n_iterations)

    def _sigma(self, iteration, n_iterations):
        return self.sigma_start * np.exp(-iteration / (n_iterations / np.log(self.sigma_start)))

    def _find_bmu(self, x):
        diff = self.weights - x
        dist = np.linalg.norm(diff, axis=2)
        return np.unravel_index(np.argmin(dist, axis=None), dist.shape)

    def train(self, data):
        # Initialiser les poids avec des images aléatoires du jeu de données
        self.weights = self._initialize_weights_from_data(data)

        for iteration in range(self.n_iterations):
            for x in data:
                bmu = self._find_bmu(x)
                learning_rate = self._learning_rate(iteration, self.n_iterations)
                sigma = self._sigma(iteration, self.n_iterations)
                g = self._neighborhood(bmu, sigma)
                diff = x - self.weights
                self.weights += learning_rate * g[..., np.newaxis] * diff

    def map_vects(self, data):
        return np.array([self._find_bmu(x) for x in data])

    def plot(self, data, labels):
        mapped = self.map_vects(data)
        plt.figure(figsize=(10, 10))
        for i, m in enumerate(mapped):
            plt.text(m[1], m[0], str(labels[i]), color=plt.cm.rainbow(labels[i] / 10),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xlim([0, self.m])
        plt.ylim([0, self.n])
        plt.show()


# Charger les données MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0

# Initialiser et entraîner la SOM
som = SOM(20, 20, 784, n_iterations=100)
som.train(x_train)

# Visualiser les résultats
som.plot(x_train, y_train)
