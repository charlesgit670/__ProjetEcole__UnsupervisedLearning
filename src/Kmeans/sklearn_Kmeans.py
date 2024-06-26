import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def label_distribution_per_cluster(kmeans, X, y, n_clusters):
    # Prédire les clusters pour les données X
    clusters = kmeans.predict(X)

    # Initialiser un tableau pour stocker la répartition des labels
    distribution = np.zeros((n_clusters, len(np.unique(y))), dtype=int)

    # Compter les labels pour chaque cluster
    for cluster in range(n_clusters):
        cluster_labels = y[clusters == cluster]
        for label in np.unique(y):
            distribution[cluster, label] = np.sum(cluster_labels == label)

    return distribution

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

# Charger les données MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Prétraiter les données: mise à plat et normalisation
X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32) / 255.0

n_clusters = 50

kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_train)
distribution = label_distribution_per_cluster(kmeans, X_test, y_test, n_clusters)
plot_latent_space(distribution, y_test)

