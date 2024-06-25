import os
from data_loader import load_mnist_data
from model import AutoEncoder
import utils
import numpy as np

# Créer un dossier pour les graphiques si nécessaire
if not os.path.exists('plots'):
    os.makedirs('plots')


def run_single_experiment(latent_dim, activation='tanh', final_activation='sigmoid', loss='binary_crossentropy'):
    # Charger et préparer les données
    x_train, x_test, y_train, y_test = load_mnist_data(normalize=True)

    # Créer et entraîner l'AutoEncoder
    autoencoder = AutoEncoder(input_dim=784, latent_dim=latent_dim, activation=activation,
                              final_activation=final_activation)
    history = autoencoder.train(x_train, x_test, epochs=50, batch_size=256, loss=loss)

    # Visualiser la perte d'entraînement
    loss_title = f'Training Loss (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss})'
    utils.plot_loss(history, title=loss_title, filename=f'plots/los_dim={latent_dim}_act={activation}.png')

    # Encoder et reconstruire les images
    encoded_imgs = autoencoder.encode(x_test)
    decoded_imgs = autoencoder.decode(encoded_imgs)

    # Visualiser les images originales et reconstruites
    comparison_title = f'Original and Reconstructed Images (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss})'
    utils.plot_comparison(x_test, decoded_imgs, latent_dim, title=comparison_title,
                          filename=f'plots/encode_decode_dim={latent_dim}_act={activation}.png')

    # Si latent_dim == 2 ou 3, visualiser l'espace latent
    if latent_dim in [2, 3]:
        latent_space_title = f'Latent Space Representation (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss})'
        utils.plot_latent_space(encoded_imgs, latent_dim, y_test, title=latent_space_title,
                                filename=f'plots/latent_space_dim={latent_dim}_act={activation}.png')

    # Générer des données synthétiques
    synthetic_data, latent_points = autoencoder.generate_synthetic_data()
    synthetic_title = f'Synthetic Data (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss})'
    utils.plot_synthetic_data(synthetic_data, latent_points, title=synthetic_title,
                              filename=f'plots/gendata_dim={latent_dim}_act={activation}.png')


def run_exploration(activations, losses, latent_dims):
    # Charger et préparer les données
    x_train, x_test, y_train, y_test = load_mnist_data(normalize=True)

    histories = {}
    total_explorations = len(activations) * len(losses) * len(latent_dims)
    current_exploration = 0

    for activation in activations:
        for loss in losses:
            for latent_dim in latent_dims:
                current_exploration += 1
                print(
                    f"Running exploration {current_exploration}/{total_explorations} - Activation: {activation}, Loss: {loss}, Latent Dim: {latent_dim}")

                autoencoder = AutoEncoder(input_dim=784, latent_dim=latent_dim, activation=activation,
                                          final_activation='sigmoid')
                history = autoencoder.train(x_train, x_test, epochs=50, batch_size=256, loss=loss)
                histories[(activation, loss, latent_dim)] = history

    # Visualiser les pertes d'entraînement pour chaque combinaison d'activation et de perte
    utils.plot_multiple_histories(histories, activations, losses, latent_dims,
                                  filename=f'plots/exploration.png')


if __name__ == "__main__":
    # Exécution d'une seule expérience
    run_single_experiment(latent_dim=3, activation='tanh', final_activation='sigmoid', loss='binary_crossentropy')

    # Exploration automatique de plusieurs configurations
    activations = ['relu', 'tanh']
    losses = ['binary_crossentropy', 'mse']
    latent_dims = [2, 3, 5]
    # run_exploration(activations, losses, latent_dims)
