import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from annotation import timer
from load_image import load_data_food


class KMeans:
    def __init__(self, n_clusters=10, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        np.random.seed(42)
        initial_centroids_idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[initial_centroids_idx]

        for i in range(self.max_iter):
            if i % 10 == 0:
                print("iteration : ", i)
            self.labels = self._assign_clusters(X)
            new_centroids = self._calculate_centroids(X)

            if np.all(np.abs(self.centroids - new_centroids) < self.tol):
                print(f"Kmeans has converged, iteration {i}")
                break

            self.centroids = new_centroids

    # @timer
    # def _assign_clusters(self, X):
    #     distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
    #     return np.argmin(distances, axis=1)

    # @timer
    def _assign_clusters(self, X):
        # Optimisation en utilisant np.einsum pour calculer les distances carrées
        X_squared = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
        centroids_squared = np.einsum('ij,ij->i', self.centroids, self.centroids)
        cross_term = np.dot(X, self.centroids.T)

        distances_squared = X_squared + centroids_squared - 2 * cross_term
        return np.argmin(distances_squared, axis=1)

    # @timer
    def _calculate_centroids(self, X):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            points_in_cluster = X[self.labels == k]
            centroids[k] = np.mean(points_in_cluster, axis=0)
        return centroids

    # @timer
    def _assign_n_closer_clusters(self, X, n=3):
        # Optimisation en utilisant np.einsum pour calculer les distances carrées
        X_squared = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
        centroids_squared = np.einsum('ij,ij->i', self.centroids, self.centroids)
        cross_term = np.dot(X, self.centroids.T)

        distances_squared = X_squared + centroids_squared - 2 * cross_term
        return np.argsort(distances_squared, axis=1)[:, :n]

    def predict(self, X):
        return self._assign_clusters(X)

    def reconstruct_image(self, X):
        closest_clusters = self._assign_n_closer_clusters(X, 3)
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        closest_distances = np.take_along_axis(distances, closest_clusters, axis=1)

        weights = 1 / (closest_distances + 1e-10)  # Inverser les distances pour obtenir des poids
        weights /= weights.sum(axis=1, keepdims=True)  # Normaliser les poids

        reconstructed = np.sum(self.centroids[closest_clusters] * weights[:, :, np.newaxis], axis=1)
        return reconstructed, closest_clusters

    def reconstruct_image_naive(self, X):
        closest_clusters = self._assign_n_closer_clusters(X, 1)
        reconstructed = self.centroids[closest_clusters]
        return reconstructed

    def label_distribution_per_cluster(self, X, y):
        # Prédire les clusters pour les données X
        clusters = self._assign_clusters(X)

        # Initialiser un tableau pour stocker la répartition des labels
        distribution = np.zeros((self.n_clusters, len(np.unique(y))), dtype=int)

        # Compter les labels pour chaque cluster
        for cluster in range(self.n_clusters):
            cluster_labels = y[clusters == cluster]
            for label in np.unique(y):
                distribution[cluster, label] = np.sum(cluster_labels == label)

        return distribution

    def generate_images_gaussian_noise(self, n_samples_per_cluster=1, std_dev=0.1):
        # Générer de nouvelles valeurs autour des centroids
        new_images = np.zeros((n_samples_per_cluster, self.centroids.shape[0], self.centroids.shape[1]))
        for i in range(n_samples_per_cluster):
            for j in range(self.centroids.shape[0]):
                new_images[i, j] = np.clip(self.centroids[j] + np.random.normal(0, std_dev, self.centroids.shape[1]), 0, 1)

        return new_images

    def generate_weighted_images(self, weight_factor=2.0):
        # Générer de nouvelles valeurs comme une combinaison linéaire pondérée des centroids
        new_images = np.zeros((self.n_clusters, self.centroids.shape[1]))
        for i in range(self.n_clusters):
            weights = np.ones(self.n_clusters) / (self.n_clusters + weight_factor - 1)
            weights[i] *= weight_factor
            new_images[i] = np.dot(weights, self.centroids)
        return new_images

def plot_latent_space(distribution, y):
    n_clusters = distribution.shape[0]
    n_cols = 5
    n_rows = (n_clusters + n_cols - 1) // n_cols  # Calculer le nombre de lignes nécessaires

    # Afficher la répartition sous forme d'histogramme
    plt.figure(figsize=(15, 10))
    for i in range(n_clusters):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.bar(np.unique(y), distribution[i])
        plt.title(f'Cluster {i}')
        plt.xlabel('Label')
        plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_reconstructed_image(X_test, predictions, reconstructed_images_naive, reconstructed_images, clusters, shapex, shapey, shapez):
        # Affichage des résultats pour quelques images de test
    plt.figure(figsize=(20, 10))  # Augmenter la taille de la figure pour plus de lisibilité
    for i in range(10):  # Limiter à 10 images pour éviter la surcharge
        # Affichage des images originales
        plt.subplot(3, 10, i + 1)
        plt.imshow(X_test[i].reshape(shapex, shapey, shapez), cmap='gray')
        plt.title(f'O:{i}', fontsize=15)
        plt.axis('off')

        # Affichage des premières images reconstruites
        plt.subplot(3, 10, i + 11)
        plt.imshow(reconstructed_images_naive[i].reshape(shapex, shapey, shapez), cmap='gray')
        plt.title(f'R1, C:{predictions[i]}', fontsize=15)
        plt.axis('off')

        # Affichage des deuxièmes images reconstruites
        plt.subplot(3, 10, i + 21)
        plt.imshow(reconstructed_images[i].reshape(shapex, shapey, shapez), cmap='gray')
        plt.title(f'R2, C:{clusters[i]}', fontsize=15)
        plt.axis('off')

    plt.tight_layout()  # Pour ajuster l'espacement automatiquement
    plt.show()

def plot_generated_images(generated_images, n_clusters, n_samples_per_cluster, shapex, shapey, shapez):
    rows = n_clusters
    cols = n_samples_per_cluster

    plt.figure(figsize=(15, rows * 2))  # Ajuster la taille de la figure en fonction du nombre de lignes
    for i in range(n_clusters):
        for j in range(n_samples_per_cluster):
            index = i * n_samples_per_cluster + j
            plt.subplot(rows, cols, index + 1)
            plt.imshow(generated_images[j, i].reshape(shapex, shapey, shapez), cmap='gray')
            plt.title(f'Cluster {i}, Sample {j}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Charger les données MNIST
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    shapex = 28
    shapey = 28
    shapez = 1

    # Charger les données food-101
    # X_train, X_test, y_train, y_test, label_to_classname = load_data_food()
    # shapex = 32
    # shapey = 32
    # shapez = 3

    # Prétraiter les données: mise à plat et normalisation
    X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32) / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32) / 255.0

    n_clusters = 30

    # Initialiser et entraîner le modèle k-means
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_train)

    # plot reconstructed images
    predictions = kmeans.predict(X_test)
    reconstructed_images_naive = reconstructed_images_naive = kmeans.reconstruct_image_naive(X_test)
    reconstructed_images, clusters = kmeans.reconstruct_image(X_test)
    plot_reconstructed_image(X_test, predictions, reconstructed_images_naive, reconstructed_images, clusters, shapex, shapey, shapez)

    # plot latent space
    distribution = kmeans.label_distribution_per_cluster(X_test, y_test)
    plot_latent_space(distribution, y_test)

    # plot generated image gaussian noise
    n_samples_per_cluster = 5
    generated_images = kmeans.generate_images_gaussian_noise(n_samples_per_cluster, 0.1)
    plot_generated_images(generated_images, n_clusters, n_samples_per_cluster, shapex, shapey, shapez)

    # plot generated image weights clusters
    generated_images = kmeans.generate_weighted_images(weight_factor=n_clusters)
    plot_generated_images(generated_images.reshape(1, n_clusters, -1), n_clusters, 1, shapex, shapey, shapez)

