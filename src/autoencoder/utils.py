import os
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


def plot_comparison(x_test, decoded_imgs, latent_dim, title, filename=None, n=10, image_shape=(28, 28)):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Affichage des images originales
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(image_shape))
        plt.gray()
        ax.set_title('Original')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Affichage des images reconstruites
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(image_shape))
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


def plot_synthetic_data(synthetic_data, latent_points, title, filename=None, step=0.25, image_shape=(28, 28)):
    num_images = len(synthetic_data)
    n_cols = 5  # 5 valeurs différentes pour chaque dimension
    n_rows = num_images // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(synthetic_data[i].reshape(image_shape), cmap='gray')
        x, y, z = latent_points[i]
        ax.set_title(f'({x:.2f}, {y:.2f}, {z:.2f})', fontsize=8)
        ax.axis('off')

    for ax in axes[num_images:]:
        ax.axis('off')

    plt.suptitle(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_multiple_histories(histories, activations, losses, latent_dims, learning_rates, save_dir=None, dataset=''):
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    for loss in losses:
        for lr in learning_rates:
            fig, axs = plt.subplots(len(latent_dims), len(activations), figsize=(15, 10))
            for i, activation in enumerate(activations):
                for j, latent_dim in enumerate(latent_dims):
                    history = histories[(activation, loss, latent_dim, lr)]
                    style = styles[j % len(styles)]
                    marker = markers[j % len(markers)]
                    axs[j, i].plot(history.history['loss'], label='Train Loss', linestyle=style, marker=marker)
                    axs[j, i].plot(history.history['val_loss'], label='Val Loss', linestyle=style, marker=marker)
                    axs[j, i].set_title(f'Act: {activation}, Latent Dim: {latent_dim}, LR: {lr}')
                    axs[j, i].set_xlabel('Epochs')
                    axs[j, i].set_ylabel('Loss')
                    axs[j, i].legend()

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Ajuster les marges pour inclure les titres et les labels
            fig.suptitle(f'Training and Validation Loss for Loss Function: {loss}', fontsize=16)

            if save_dir:
                filename = os.path.join(save_dir, f'{dataset}_exploration_{loss}_lr={lr}.png')
                plt.savefig(filename, bbox_inches='tight')

            plt.show()
