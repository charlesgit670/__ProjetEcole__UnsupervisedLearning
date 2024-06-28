import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D


def get_loss_function(loss_name):
    """Get the loss function by name."""
    if loss_name == 'binary_crossentropy':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss_name == 'mse':
        return tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def plot_loss(history, title="Training Loss", filename=None):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.show()


def plot_latent_space(vae, x_test, y_test, latent_dim, n=30, figsize=15, filename=None):
    """
    Plots the latent space for 2D or 3D latent dimensions.

    Parameters:
    vae (VAE): The trained VAE model.
    x_test (np.array): Test data.
    y_test (np.array): Test labels.
    latent_dim (int): The dimension of the latent space.
    n (int): Number of points along each axis.
    figsize (int): Size of the plot.
    filename (str): File name to save the plot.
    """
    if latent_dim == 2:
        # Display a 2D manifold of digits
        digit_size = 28
        scale = 1.0
        figure = np.zeros((digit_size * n, digit_size * n))
        grid_x = np.linspace(-scale, scale, n)
        grid_y = np.linspace(-scale, scale, n)[::-1]

        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                x_decoded = vae.decoder.predict(z_sample, verbose=0)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[
                i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size,
                ] = digit

        plt.figure(figsize=(figsize, figsize))
        start_range = digit_size // 2
        end_range = n * digit_size + start_range
        pixel_range = np.arange(start_range, end_range, digit_size)
        sample_range_x = np.round(grid_x, 1)
        sample_range_y = np.round(grid_y, 1)
        plt.xticks(pixel_range, sample_range_x)
        plt.yticks(pixel_range, sample_range_y)
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.imshow(figure, cmap="Greys_r")
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()

    elif latent_dim == 3:
        encoded_imgs = vae.encoder.predict(x_test)[0]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], encoded_imgs[:, 2], c=y_test, cmap='viridis')
        plt.colorbar(sc)
        ax.set_xlabel('Latent Dimension 1')
        ax.set_ylabel('Latent Dimension 2')
        ax.set_zlabel('Latent Dimension 3')
        plt.title(f'Latent Space Representation (latent_dim={latent_dim})')
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        plt.show()


def plot_reconstructions(model, x_test, title="Reconstructions", filename=None):
    decoded_imgs = model.predict(x_test)
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
        ax.axis("off")

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_synthetic_data(synthetic_data, latent_points, title, filename=None):
    num_images = len(synthetic_data)
    n_cols = 5  # 5 valeurs différentes pour chaque dimension
    n_rows = (num_images + n_cols - 1) // n_cols  # Ajustement du nombre de lignes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()

    for i in range(num_images):
        ax = axes[i]
        image = synthetic_data[i].numpy()  # Convertir le tenseur en tableau NumPy
        ax.imshow(image.reshape(28, 28), cmap='gray')
        coords = ", ".join([f"{coord:.2f}" for coord in latent_points[i][:2]])  # Utiliser les deux premières dimensions du point latent pour l'affichage
        ax.set_title(f'({coords})', fontsize=8)
        ax.axis('off')

    for ax in axes[num_images:]:
        ax.axis('off')

    plt.suptitle(title)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_multiple_histories(histories, latent_dims):
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']

    for latent_dim in latent_dims:
        fig, ax = plt.subplots(figsize=(10, 6))
        history = histories[latent_dim]
        ax.plot(history.history['loss'], label='Train Loss', linestyle=styles[0], marker=markers[0])
        ax.plot(history.history['val_loss'], label='Val Loss', linestyle=styles[1], marker=markers[1])
        ax.set_title(f'Latent Dim: {latent_dim}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.suptitle(f'Training and Validation Loss for Latent Dim: {latent_dim}', fontsize=16)
        plt.savefig(f'plots/exploration_loss_dim={latent_dim}.png', bbox_inches='tight')
        plt.show()
