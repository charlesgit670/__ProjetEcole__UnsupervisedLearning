from explore_data import load_mnist_data, visualize_data
from vae import VAE
from evaluate_vae import evaluate_vae
from visualize_results import plot_loss, plot_comparison, plot_latent_space, plot_synthetic_data, plot_multiple_histories
import os

def run_single_experiment(latent_dim=3, activation='tanh', final_activation='sigmoid', loss='binary_crossentropy', epochs=50):
    # Charger et visualiser les données
    (x_train, y_train), (x_test, y_test) = load_mnist_data(normalize=True)
    # visualize_data(x_train)

    # Initialiser et entraîner le VAE
    vae = VAE(input_shape=x_train.shape[1:], latent_dim=latent_dim, activation=activation, final_activation=final_activation)
    history = vae.train(x_train, batch_size=128, epochs=epochs, loss=loss)
    vae.save_models()

    # Créer le dossier plots si nécessaire
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Visualiser la perte pendant l'entraînement
    plot_loss(history, title=f"Training Loss (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss})", filename='plots/training_loss.png')

    # Visualiser les images originales et reconstruites
    x_test_decoded = vae.vae.predict(x_test)
    plot_comparison(x_test, x_test_decoded, latent_dim, title=f"Original and Reconstructed Images (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss})", filename='plots/reconstructed_images.png')

    # Évaluer le VAE
    evaluate_vae('mnist', latent_dim)

    # Générer des données synthétiques
    synthetic_data, latent_points = vae.generate_synthetic_data()
    plot_synthetic_data(synthetic_data, latent_points, title=f"Synthetic Data (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss})", filename='plots/synthetic_data.png')

def run_exploration(activations, losses, latent_dims, epochs=50):
    histories = {}
    for activation in activations:
        for loss in losses:
            for latent_dim in latent_dims:
                print(f"Running experiment with activation={activation}, loss={loss}, latent_dim={latent_dim}")
                histories[(activation, loss, latent_dim)] = run_single_experiment(latent_dim=latent_dim, activation=activation, final_activation='sigmoid', loss=loss, epochs=epochs)
    plot_multiple_histories(histories, activations, losses, latent_dims, filename_prefix='plots/exploration_')

if __name__ == "__main__":
    # Exécution d'une seule expérience
    run_single_experiment(latent_dim=3, activation='tanh', final_activation='sigmoid', loss='binary_crossentropy', epochs=3)

    # Exploration automatique de plusieurs configurations
    activations = ['relu', 'tanh']
    losses = ['binary_crossentropy', 'mse']
    latent_dims = [3, 30]
    # run_exploration(activations, losses, latent_dims, epochs=5)
