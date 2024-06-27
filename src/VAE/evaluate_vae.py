import numpy as np
from explore_data import load_mnist_data
from vae import VAE
from visualize_results import plot_latent_space


def evaluate_vae(dataset, latent_dim):
    """
    Évalue le VAE en visualisant l'espace latent et en générant des données synthétiques.
    """
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    vae = VAE(input_shape=x_train.shape[1:], latent_dim=latent_dim)
    vae.load_models()

    # Encoder les données test
    z_mean, _, _ = vae.encoder.predict(x_test, batch_size=128)

    # Visualiser l'espace latent
    if latent_dim == 2 or latent_dim == 3:
        plot_latent_space(z_mean, latent_dim, y_test, title=f"Latent Space (latent_dim={latent_dim})",
                          filename=f'plots/latent_space_dim={latent_dim}.png')
    else:
        print("Latent dimension not supported for visualization")
