import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model

class AutoEncoder:
    def __init__(self, input_dim, latent_dim, activation='relu', final_activation='sigmoid'):
        """
        Initialise l'AutoEncoder avec les dimensions spécifiées et l'activation.

        Parameters:
        input_dim (int): La dimension des données d'entrée (par exemple, 784 pour MNIST).
        latent_dim (int): La dimension de l'espace latent (taille de la compression).
        activation (str): La fonction d'activation à utiliser dans les couches (par défaut 'relu').
        final_activation (str): La fonction d'activation à utiliser dans la dernière couche (par défaut 'sigmoid').
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.activation = activation
        self.final_activation = final_activation
        self.autoencoder = self.build_autoencoder()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_autoencoder(self):
        """
        Construit l'architecture complète de l'AutoEncoder.

        Le modèle d'AutoEncoder se compose d'un encodeur et d'un décodeur.
        L'encodeur réduit la dimension des données d'entrée pour créer une représentation latente,
        tandis que le décodeur reconstruit les données d'origine à partir de cette représentation latente.
        """
        # Entrée de l'image de taille input_dim
        input_img = Input(shape=(self.input_dim,))

        # Encoder : Réduction de la dimension
        encoded = Dense(256, activation=self.activation, name="encoder_dense_1")(input_img)
        encoded = BatchNormalization(name="encoder_bn_1")(encoded)
        encoded = Dense(128, activation=self.activation, name="encoder_dense_2")(encoded)
        encoded = BatchNormalization(name="encoder_bn_2")(encoded)
        encoded = Dense(64, activation=self.activation, name="encoder_dense_3")(encoded)
        encoded = BatchNormalization(name="encoder_bn_3")(encoded)
        # La troisième couche encodée réduit à la dimension latente avec une activation spécifiée
        encoded = Dense(self.latent_dim, activation=self.activation, name="latent")(encoded)

        # Decoder : Reconstruction de l'image
        decoded = Dense(64, activation=self.activation, name="decoder_dense_1")(encoded)
        decoded = BatchNormalization(name="decoder_bn_1")(decoded)
        decoded = Dense(128, activation=self.activation, name="decoder_dense_2")(decoded)
        decoded = BatchNormalization(name="decoder_bn_2")(decoded)
        decoded = Dense(256, activation=self.activation, name="decoder_dense_3")(decoded)
        decoded = BatchNormalization(name="decoder_bn_3")(decoded)
        # La couche de sortie reconstruit les données d'entrée avec une activation spécifiée
        decoded = Dense(self.input_dim, activation=self.final_activation, name="output")(decoded)

        autoencoder = Model(input_img, decoded)

        return autoencoder

    def build_encoder(self):
        """
        Construit l'architecture de l'encodeur seule.
        L'encodeur prend l'entrée de l'AutoEncoder et sort la couche encodée (latente)
        """
        return Model(self.autoencoder.input, self.autoencoder.get_layer("latent").output)

    def build_decoder(self):
        """
        Construit l'architecture du décodeur seule.
        Le décodeur prend la représentation latente et reconstruit les données d'origine.
        """
        encoded_input = Input(shape=(self.latent_dim,))
        x = encoded_input
        for layer_name in ["decoder_dense_1", "decoder_bn_1", "decoder_dense_2", "decoder_bn_2", "decoder_dense_3", "decoder_bn_3", "output"]:
            x = self.autoencoder.get_layer(layer_name)(x)

        return Model(encoded_input, x)

    def train(self, x_train, x_test, epochs=300, batch_size=256, loss='binary_crossentropy', learning_rate=0.001):
        """
        Entraîne l'AutoEncoder avec les données de formation.

        Parameters:
        x_train (ndarray): Les données de formation.
        x_test (ndarray): Les données de test.
        epochs (int): Le nombre d'époques pour l'entraînement.
        batch_size (int): La taille du batch pour l'entraînement.
        loss (str): La fonction de perte à utiliser.
        learning_rate (float): Le taux d'apprentissage pour l'optimiseur.

        Returns:
        history (History): L'historique de l'entraînement contenant les pertes pour chaque époque.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        return self.autoencoder.fit(x_train, x_train,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    validation_data=(x_test, x_test))

    def encode(self, x):
        """
        Encode les données d'entrée en utilisant l'encodeur.

        Parameters:
        x (ndarray): Les données d'entrée à encoder.

        Returns:
        encoded (ndarray): Les données encodées (réduites) dans l'espace latent.
        """
        return self.encoder.predict(x)

    def decode(self, encoded_imgs):
        """
        Décode les données encodées en utilisant le décodeur.

        Parameters:
        encoded_imgs (ndarray): Les données encodées à décoder.

        Returns:
        decoded (ndarray): Les données reconstruites à partir de l'espace latent.
        """
        return self.decoder.predict(encoded_imgs)

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
        synthetic_data = self.decode(latent_points)
        return synthetic_data, latent_points
