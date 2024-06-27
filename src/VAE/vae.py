import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Conv2D, Flatten, Conv2DTranspose, Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
import numpy as np
import os

@tf.keras.utils.register_keras_serializable()
def sampling(args):
    """
    Trick de reparamétrisation pour assurer la différentiabilité.
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

@tf.keras.utils.register_keras_serializable()
class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, x_decoded_mean, z_mean, z_log_var = inputs
        reconstruction_loss = tf.reduce_mean(mse(tf.keras.backend.flatten(x), tf.keras.backend.flatten(x_decoded_mean)))
        reconstruction_loss *= x.shape[1] * x.shape[2]
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class VAE:
    def __init__(self, input_shape, latent_dim, activation='relu', final_activation='sigmoid'):
        """
        Initialise le VAE avec les dimensions spécifiées et l'activation.

        Parameters:
        input_shape (tuple): La dimension des données d'entrée (par exemple, (28, 28, 1) pour MNIST).
        latent_dim (int): La dimension de l'espace latent.
        activation (str): La fonction d'activation à utiliser dans les couches (par défaut 'relu').
        final_activation (str): La fonction d'activation à utiliser dans la dernière couche (par défaut 'sigmoid').
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.activation = activation
        self.final_activation = final_activation
        self.encoder = None
        self.decoder = None
        self.vae = None
        self._build()

    def _build(self):
        """
        Construit le modèle VAE : encodeur, décodeur et VAE complet.
        """
        # Construction de l'encodeur
        inputs = Input(shape=self.input_shape)  # Entrée de l'image de taille input_shape
        x = Conv2D(32, 3, activation=self.activation, padding='same')(inputs)  # Convolution avec 32 filtres et activation
        x = Conv2D(64, 3, activation=self.activation, padding='same', strides=(2, 2))(x)  # Réduction de la taille avec stride
        x = Conv2D(64, 3, activation=self.activation, padding='same')(x)  # Convolution supplémentaire
        x = Conv2D(64, 3, activation=self.activation, padding='same')(x)  # Convolution supplémentaire
        shape = tf.keras.backend.int_shape(x)  # Stockage de la forme pour la phase de décodeur

        x = Flatten()(x)  # Aplatissement de la matrice
        x = Dense(32, activation=self.activation)(x)  # Couche dense intermédiaire
        z_mean = Dense(self.latent_dim)(x)  # Calcul de la moyenne de z
        z_log_var = Dense(self.latent_dim)(x)  # Calcul de la variance de z

        # Utilisation du trick de reparamétrisation
        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])  # Échantillonnage de z

        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')  # Création du modèle encodeur
        encoder.summary()

        # Construction du décodeur
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')  # Entrée de l'espace latent
        x = Dense(shape[1] * shape[2] * shape[3], activation=self.activation)(latent_inputs)  # Couche dense initiale
        x = Reshape((shape[1], shape[2], shape[3]))(x)  # Reshape aux dimensions de l'encodeur
        x = Conv2DTranspose(64, 3, activation=self.activation, padding='same')(x)  # Convolution transpose
        x = Conv2DTranspose(64, 3, activation=self.activation, padding='same')(x)  # Convolution transpose
        x = Conv2DTranspose(32, 3, activation=self.activation, padding='same', strides=(2, 2))(x)  # Augmentation de la taille
        outputs = Conv2DTranspose(1, 3, activation=self.final_activation, padding='same')(x)  # Sortie finale

        decoder = Model(latent_inputs, outputs, name='decoder')  # Création du modèle décodeur
        decoder.summary()

        # Construction du VAE complet
        outputs = decoder(encoder(inputs)[2])  # Sortie du décodeur
        vae_loss_layer = VAELossLayer()([inputs, outputs, encoder(inputs)[0], encoder(inputs)[1]])  # Calcul de la perte VAE
        self.vae = Model(inputs, vae_loss_layer)  # Création du modèle VAE complet

        self.encoder = encoder
        self.decoder = decoder
        self.vae.compile(optimizer='adam')  # Compilation du modèle

    def train(self, x_train, batch_size=128, epochs=50, loss='binary_crossentropy'):
        """
        Entraînement du modèle VAE.
        """
        history = self.vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        return history

    def save_models(self, model_dir='models', encoder_path='encoder.h5', decoder_path='decoder.h5', vae_path='vae.h5'):
        """
        Sauvegarde des modèles entraînés.
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.encoder.save(os.path.join(model_dir, encoder_path))
        self.decoder.save(os.path.join(model_dir, decoder_path))
        self.vae.save(os.path.join(model_dir, vae_path))

    def load_models(self, model_dir='models', encoder_path='encoder.h5', decoder_path='decoder.h5', vae_path='vae.h5'):
        """
        Chargement des modèles sauvegardés.
        """
        custom_objects = {'sampling': sampling, 'VAELossLayer': VAELossLayer}
        self.encoder = tf.keras.models.load_model(os.path.join(model_dir, encoder_path), custom_objects=custom_objects, compile=False)
        self.decoder = tf.keras.models.load_model(os.path.join(model_dir, decoder_path), custom_objects=custom_objects, compile=False)
        self.vae = tf.keras.models.load_model(os.path.join(model_dir, vae_path), custom_objects=custom_objects, compile=False)

    def generate_synthetic_data(self, step=0.25):
        """
        Génère des données en échantillonnant l'espace latent avec des pas de 0.25.

        Parameters:
        step (float): Le pas de l'échantillonnage dans l'espace latent.

        Returns:
        synthetic_data (ndarray): Les données synthétiques générées.
        """
        ranges = np.arange(0, 1 + step, step)
        latent_points = np.array([[x, y, z] for x in ranges for y in ranges for z in ranges])
        synthetic_data = self.decoder.predict(latent_points)
        return synthetic_data, latent_points
