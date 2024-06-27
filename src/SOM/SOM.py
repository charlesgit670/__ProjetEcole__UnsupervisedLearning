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

    def plot_label_histograms_per_cluster(self, data, labels):
        mapped = self.map_vects(data)
        label_counts = np.zeros((self.m, self.n, 10))

        for i, m in enumerate(mapped):
            label_counts[m[0], m[1], labels[i]] += 1

        plt.figure(figsize=(25, 25))
        for i in range(self.m):
            for j in range(self.n):
                plt.subplot(self.m, self.n, i * self.n + j + 1)
                plt.bar(range(10), label_counts[i, j])
                plt.xticks(range(10))
                plt.yticks([])
                plt.title(f'({i},{j})')
        plt.show()

    def plot_reconstructed_images(self, data, labels, gamma):
        fig, axes = plt.subplots(3, 10, figsize=(20, 6))

        # Afficher un exemple par label et son neurone BMU
        unique_labels = np.unique(labels)
        weights_flatten = self.weights.reshape(self.n * self.m, data.shape[1])
        for label in unique_labels:
            indices = np.where(labels == label)[0]

            example_index = indices[0]
            example_image = data[example_index]
            bmu = self._find_bmu(example_image)
            bmu_image = self.weights[bmu].reshape(28, 28)

            axes[0, label].imshow(example_image.reshape(28, 28), cmap='gray')
            axes[0, label].axis('off')
            axes[0, label].set_title(f'Original {label}')

            axes[1, label].imshow(bmu_image, cmap='gray')
            axes[1, label].axis('off')
            axes[1, label].set_title(f'BMU {label}')

            # Calculer l'image combinée
            distances = np.linalg.norm(example_image - weights_flatten, axis=1)
            ponderations = np.exp(-distances ** 2 / gamma)
            ponderations /= ponderations.sum(keepdims=True)
            combined_image = np.dot(weights_flatten.T, ponderations)

            axes[2, label].imshow(combined_image.reshape(28, 28), cmap='gray')
            axes[2, label].axis('off')
            axes[2, label].set_title('Combined Image')

        plt.tight_layout()
        plt.show()

    def interpolate(self, x, y):
        # Calculer les coordonnées des quatre neurones les plus proches
        x1, y1 = int(np.floor(x)), int(np.floor(y))
        x2, y2 = min(x1 + 1, self.m - 1), min(y1 + 1, self.n - 1)

        # Calculer les distances fractionnelles
        dx, dy = x - x1, y - y1

        # Obtenir les poids des neurones
        w11 = self.weights[x1, y1]
        w12 = self.weights[x1, y2]
        w21 = self.weights[x2, y1]
        w22 = self.weights[x2, y2]

        # Interpolation bilinéaire
        interpolated_weight = (
                (1 - dx) * (1 - dy) * w11 +
                (1 - dx) * dy * w12 +
                dx * (1 - dy) * w21 +
                dx * dy * w22
        )

        return interpolated_weight.reshape(28, 28)

if __name__ == "__main__":
    # Charger les données MNIST
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Prétraiter les données: mise à plat et normalisation
    X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32) / 255.0

    # Initialiser et entraîner la SOM
    som = SOM(10, 10, learning_rate=0.1, gamma=1.5, n_iterations=10000)
    som.train(X_train)

    # Affiche la représentation de la map (espace latent)
    # som.plot_weights()

    # Affiche l'espace latent sous forme d'histogramme de label par cluster
    # som.plot_label_histograms_per_cluster(X_test, y_test)

    # Compression/décompression
    # som.plot_reconstructed_images(X_test, y_test, 20)

    # Génération à partir de l'interpolation entre 3 clusters
    som.interpolate(2.5, 5.6)