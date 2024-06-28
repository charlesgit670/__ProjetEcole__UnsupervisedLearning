import os
import tensorflow as tf
from src.VAE.old.data_loader import load_mnist
from model import VariationalAutoEncoder
import utils

# Créer un dossier pour les graphiques si nécessaire
if not os.path.exists('plots'):
    os.makedirs('plots')

def run_single_experiment(latent_dim, activation='relu', loss='binary_crossentropy', epochs=50):
    """Run a single experiment with the given parameters."""
    # Charger et préparer les données
    (x_train, _), (x_test, y_test) = load_mnist()
    input_shape = (28, 28, 1)

    # Créer et entraîner le VAE
    vae = VariationalAutoEncoder(input_shape=input_shape, latent_dim=latent_dim, activation=activation)
    loss_function = utils.get_loss_function(loss)
    vae.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_function)
    history = vae.fit_model(x_train, x_test, epochs=epochs, batch_size=256)

    # Visualiser la perte d'entraînement
    loss_title = f'Training Loss (latent_dim={latent_dim}, activation={activation}, loss={loss})'
    utils.plot_loss(history, title=loss_title, filename=f'plots/loss_dim={latent_dim}_act={activation}_loss={loss}.png')

    # Encoder et reconstruire les images
    encoded_imgs = vae.encode(x_test)
    decoded_imgs = vae.decode(encoded_imgs)

    # Visualiser les images originales et reconstruites
    comparison_title = f'Original and Reconstructed Images (latent_dim={latent_dim}, activation={activation}, loss={loss})'
    utils.plot_comparison(x_test, decoded_imgs, latent_dim, title=comparison_title,
                          filename=f'plots/encode_decode_dim={latent_dim}_act={activation}_loss={loss}.png')

    # Si latent_dim == 2 ou 3, visualiser l'espace latent
    if latent_dim in [2, 3]:
        latent_space_title = f'Latent Space Representation (latent_dim={latent_dim}, activation={activation}, loss={loss})'
        utils.plot_latent_space(encoded_imgs, latent_dim, y_test, title=latent_space_title,
                                filename=f'plots/latent_space_dim={latent_dim}_act={activation}_loss={loss}.png')

    # Générer des données synthétiques
    synthetic_data, latent_points = vae.generate_synthetic_data()
    synthetic_title = f'Synthetic Data (latent_dim={latent_dim}, activation={activation}, loss={loss})'
    utils.plot_synthetic_data(synthetic_data, latent_points, title=synthetic_title,
                              filename=f'plots/gendata_dim={latent_dim}_act={activation}_loss={loss}.png')

    # Print sample of encoded values for debugging
    print("Sample of encoded values:\n", encoded_imgs[:5])

def run_exploration(activations, losses, latent_dims):
    """Run an automatic exploration with given parameters."""
    # Charger et préparer les données
    (x_train, _), (x_test, y_test) = load_mnist()
    input_shape = (28, 28, 1)

    histories = {}
    total_explorations = len(activations) * len(losses) * len(latent_dims)
    current_exploration = 0

    for activation in activations:
        for loss in losses:
            for latent_dim in latent_dims:
                current_exploration += 1
                print(f"Running exploration {current_exploration}/{total_explorations} - Activation: {activation}, Loss: {loss}, Latent Dim: {latent_dim}")

                vae = VariationalAutoEncoder(input_shape=input_shape, latent_dim=latent_dim, activation=activation)
                loss_function = utils.get_loss_function(loss)
                vae.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss_function)
                history = vae.fit_model(x_train, x_test, epochs=10, batch_size=256)
                histories[(activation, loss, latent_dim)] = history

    # Visualiser les pertes d'entraînement pour chaque combinaison d'activation et de perte
    utils.plot_multiple_histories(histories, activations, losses, latent_dims)

if __name__ == "__main__":
    # Exécution d'une seule expérience
    run_single_experiment(latent_dim=3, activation='tanh', loss='mse', epochs=3)

    # Exploration automatique de plusieurs configurations
    activations = ['relu', 'tanh']
    losses = ['binary_crossentropy', 'mse']
    latent_dims = [3, 30]
    # run_exploration(activations, losses, latent_dims)
