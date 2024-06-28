import os
import tensorflow as tf
import numpy as np
from data_loader import load_mnist
from model import create_vae
import utils

# Créer un dossier pour les graphiques si nécessaire
if not os.path.exists('plots'):
    os.makedirs('plots')

def run_single_experiment(latent_dim=2, epochs=10):
    """Run a single experiment with the given parameters."""
    # Charger et préparer les données
    (x_train, y_train), (x_test, y_test) = load_mnist()
    assert x_train is not None and x_test is not None, "Training and test data cannot be None"

    # Vérification des données
    assert np.all(x_train is not None), "Training data contains None values"
    assert np.all(x_test is not None), "Test data contains None values"
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

    # Créer et entraîner le VAE
    vae = create_vae(latent_dim=latent_dim)
    history = vae.fit(x_train, epochs=epochs, batch_size=128, validation_data=(x_test, x_test))

    # Visualiser la perte d'entraînement
    loss_title = f'Training Loss (latent_dim={latent_dim})'
    utils.plot_loss(history, title=loss_title, filename=f'plots/loss_dim={latent_dim}.png')

    # Visualiser l'espace latent
    utils.plot_latent_space(vae, filename=f'plots/latent_space_dim={latent_dim}.png')

    # Visualiser les images originales et reconstruites
    utils.plot_reconstructions(vae, x_test, filename=f'plots/reconstructions_dim={latent_dim}.png')

    # Générer des données synthétiques
    synthetic_data, latent_points = vae.generate_synthetic_data(latent_dim=latent_dim)
    synthetic_title = f'Synthetic Data (latent_dim={latent_dim})'
    utils.plot_synthetic_data(synthetic_data, latent_points, title=synthetic_title,
                              filename=f'plots/gendata_dim={latent_dim}.png')

    # Print sample of encoded values for debugging
    encoded_imgs = vae.encoder.predict(x_test)[0]
    print("Sample of encoded values:\n", encoded_imgs[:5])

def run_exploration(latent_dims):
    """Run an automatic exploration with given parameters."""
    # Charger et préparer les données
    (x_train, y_train), (x_test, y_test) = load_mnist()
    assert x_train is not None and x_test is not None, "Training and test data cannot be None"

    # Vérification des données
    assert np.all(x_train is not None), "Training data contains None values"
    assert np.all(x_test is not None), "Test data contains None values"
    print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")

    histories = {}
    total_explorations = len(latent_dims)
    current_exploration = 0

    for latent_dim in latent_dims:
        current_exploration += 1
        print(f"Running exploration {current_exploration}/{total_explorations} - Latent Dim: {latent_dim}")

        vae = create_vae(latent_dim=latent_dim)
        history = vae.fit(x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))
        histories[latent_dim] = history

    # Visualiser les pertes d'entraînement pour chaque combinaison de dimensions latentes
    utils.plot_multiple_histories(histories, latent_dims)

if __name__ == "__main__":
    # Exécution d'une seule expérience
    run_single_experiment(latent_dim=2, epochs=10)

    # Exploration automatique de plusieurs configurations
    latent_dims = [2, 10]
    run_exploration(latent_dims)
