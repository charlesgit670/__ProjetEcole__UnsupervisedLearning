import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm


class SOM:
    def __init__(self, m, n, learning_rate, gamma, n_iterations=100, ):
        self.m = m
        self.n = n
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.locations = np.array(list(self._neuron_locations(m, n)))

    def _initialize_weights_from_data(self, data):
        indices = np.random.choice(range(data.shape[0]), size=(self.m, self.n))
        return data[indices]

    def _neuron_locations(self, m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def _neighborhood(self, center, gamma):
        d = 2 * gamma
        a = np.exp(-np.power(np.linalg.norm(self.locations - center, axis=1), 2) / d)
        return a.reshape((self.m, self.n))

    # def _learning_rate(self, iteration, n_iterations):
    #     return self.learning_rate * np.exp(-iteration / n_iterations)

    # def _gamma(self, iteration, n_iterations):
    #     return self.gamma * np.exp(-iteration / (n_iterations / np.log(self.gamma)))

    def _find_bmu(self, x):
        diff = self.weights - x
        dist = np.linalg.norm(diff, axis=2)
        return np.unravel_index(np.argmin(dist, axis=None), dist.shape)

    def train(self, data):
        # Initialiser les poids avec des images aléatoires du jeu de données
        self.weights = self._initialize_weights_from_data(data)

        for _ in tqdm(range(self.n_iterations)):
            # Sélectionner une donnée au hasard
            random_index = np.random.randint(0, data.shape[0])
            x = data[random_index]

            bmu = self._find_bmu(x)
            # learning_rate = self._learning_rate(iteration, self.n_iterations)
            # gamma = self._(iteration, self.n_iterations)
            g = self._neighborhood(bmu, self.gamma)
            diff = x - self.weights
            self.weights += self.learning_rate * g[..., np.newaxis] * diff
        print("Train ended")

    def map_vects(self, data):
        return np.array([self._find_bmu(x) for x in data])

    def plot_weights(self):
        plt.figure(figsize=(10, 10))
        for i in range(self.m):
            for j in range(self.n):
                plt.subplot(self.m, self.n, i * self.n + j + 1)
                plt.imshow(self.weights[i, j].reshape(28, 28), cmap='gray')
                plt.axis('off')
        plt.show()

    # def plot(self, data, labels):
    #     mapped = self.map_vects(data)
    #     plt.figure(figsize=(10, 10))
    #     for i, m in enumerate(mapped):
    #         plt.text(m[1], m[0], str(labels[i]), color=plt.cm.rainbow(labels[i] / 10),
    #                  fontdict={'weight': 'bold', 'size': 9})
    #     plt.xlim([0, self.m])
    #     plt.ylim([0, self.n])
    #     plt.show()

if __name__ == "__main__":
    # Charger les données MNIST
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Prétraiter les données: mise à plat et normalisation
    X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32) / 255.0

    # Initialiser et entraîner la SOM
    som = SOM(10, 10, learning_rate=0.1, gamma=0.5, n_iterations=10000)
    som.train(X_train)

    som.plot_weights()
    # Visualiser les résultats
    # som.plot(X_train, y_train)
    # som.plot(X_test, y_test)
