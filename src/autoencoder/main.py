import os
from data_loader import load_mnist_data, load_food_data
from model import AutoEncoder
import utils

# Créer un dossier pour les graphiques si nécessaire
if not os.path.exists('plots'):
    os.makedirs('plots')

def run_single_experiment(dataset='mnist', latent_dim=30, activation='tanh', final_activation='sigmoid', loss='binary_crossentropy', learning_rate=0.001):
    # Charger et préparer les données
    if dataset == 'mnist':
        x_train, x_test, y_train, y_test = load_mnist_data(normalize=True)
        input_dim = 784
    elif dataset == 'food':
        x_train, x_test, y_train, y_test, label_to_classname = load_food_data(target_size=64)  # Mise à jour pour 64x64 images
        input_dim = 64 * 64 * 3

    # Créer et entraîner l'AutoEncoder
    autoencoder = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim, activation=activation, final_activation=final_activation)
    history = autoencoder.train(x_train, x_test, epochs=120, batch_size=256, loss=loss, learning_rate=learning_rate)

    # Visualiser la perte d'entraînement
    loss_title = f'Training Loss (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss}, lr={learning_rate})'
    utils.plot_loss(history, title=loss_title, filename=f'plots/{dataset}_loss_dim={latent_dim}_act={activation}_loss={loss}_lr={learning_rate}.png')

    # Encoder et reconstruire les images
    encoded_imgs = autoencoder.encode(x_test)
    decoded_imgs = autoencoder.decode(encoded_imgs)

    # Visualiser les images originales et reconstruites
    comparison_title = f'Original and Reconstructed Images (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss}, lr={learning_rate})'
    utils.plot_comparison(x_test, decoded_imgs, latent_dim, title=comparison_title, filename=f'plots/{dataset}_encode_decode_dim={latent_dim}_act={activation}_loss={loss}_lr={learning_rate}.png', image_shape=(64, 64, 3) if dataset == 'food' else (28, 28))

    # Si latent_dim == 2 ou 3, visualiser l'espace latent
    if latent_dim in [2, 3]:
        latent_space_title = f'Latent Space Representation (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss}, lr={learning_rate})'
        utils.plot_latent_space(encoded_imgs, latent_dim, y_test, title=latent_space_title, filename=f'plots/{dataset}_latent_space_dim={latent_dim}_act={activation}_loss={loss}_lr={learning_rate}.png')

    # Générer des données synthétiques
    synthetic_data, latent_points = autoencoder.generate_synthetic_data()
    synthetic_title = f'Synthetic Data (latent_dim={latent_dim}, activation={activation}, final_activation={final_activation}, loss={loss}, lr={learning_rate})'
    utils.plot_synthetic_data(synthetic_data, latent_points, title=synthetic_title, filename=f'plots/{dataset}_gendata_dim={latent_dim}_act={activation}_loss={loss}_lr={learning_rate}.png', image_shape=(64, 64, 3) if dataset == 'food' else (28, 28))

def run_exploration(activations, losses, latent_dims, learning_rates, dataset='mnist'):
    # Charger et préparer les données
    if dataset == 'mnist':
        x_train, x_test, y_train, y_test = load_mnist_data(normalize=True)
        input_dim = 784
    elif dataset == 'food':
        x_train, x_test, y_train, y_test, label_to_classname = load_food_data(target_size=64)  # Mise à jour pour 64x64 images
        input_dim = 64 * 64 * 3

    histories = {}
    total_explorations = len(activations) * len(losses) * len(latent_dims) * len(learning_rates)
    current_exploration = 0

    for activation in activations:
        for loss in losses:
            for latent_dim in latent_dims:
                for lr in learning_rates:
                    current_exploration += 1
                    print(f"Running exploration {current_exploration}/{total_explorations} - Activation: {activation}, Loss: {loss}, Latent Dim: {latent_dim}, Learning Rate: {lr}")

                    autoencoder = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim, activation=activation, final_activation='sigmoid')
                    history = autoencoder.train(x_train, x_test, epochs=300, batch_size=256, loss=loss, learning_rate=lr)
                    histories[(activation, loss, latent_dim, lr)] = history

    # Visualiser les pertes d'entraînement pour chaque combinaison d'activation et de perte
    utils.plot_multiple_histories(histories, activations, losses, latent_dims, learning_rates, save_dir='plots', dataset=dataset)

if __name__ == "__main__":
    # Exécution d'une seule expérience
    run_single_experiment(dataset='food', latent_dim=30, activation='tanh', final_activation='sigmoid', loss='binary_crossentropy', learning_rate=0.001)

    # Exploration automatique de plusieurs configurations
    activations = ['relu', 'tanh']
    losses = ['binary_crossentropy', 'mse']
    latent_dims = [3, 30]
    learning_rates = [0.001, 0.0001]
    #run_exploration(activations, losses, latent_dims, learning_rates, dataset='food')
