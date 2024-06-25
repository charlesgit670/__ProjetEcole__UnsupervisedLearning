import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Définir le backend interactif
plt.switch_backend('Qt5Agg')  # ou 'TkAgg' si vous préférez


def plot_loss(history, title="Training Loss", filename=None):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_comparison(x_test, decoded_imgs, latent_dim, title, filename=None, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Affichage des images originales
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.set_title('Original')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Affichage des images reconstruites
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.set_title('Reconstructed')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_latent_space(encoded_imgs, latent_dim, y_test, title, filename=None):
    if latent_dim == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title(title)
        if filename:
            plt.savefig(filename)
        plt.show()
    elif latent_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], encoded_imgs[:, 2], c=y_test, cmap='viridis')
        plt.colorbar(sc)
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_zlabel('Latent Dimension 3')
        plt.title(title)
        if filename:
            plt.savefig(filename)
        plt.show()


def plot_synthetic_data(synthetic_data, latent_points, title, filename=None, step=0.25):
    num_images = len(synthetic_data)
    n_cols = int(np.ceil(np.sqrt(num_images)))
    n_rows = int(np.ceil(num_images / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 2))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(synthetic_data[i].reshape(28, 28), cmap='gray')
        x, y, z = latent_points[i]
        ax.set_title(f'({x:.2f}, {y:.2f}, {z:.2f})', fontsize=10)
        ax.axis('off')

    for ax in axes[num_images:]:
        ax.axis('off')

    plt.suptitle(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


