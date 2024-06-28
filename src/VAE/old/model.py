import numpy as np
import tensorflow as tf
from keras import Input
from keras.src.layers import Conv2D, Flatten, Dense
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Layer that uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        """Perform the reparameterization trick by sampling from a normal distribution."""
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoEncoder(tf.keras.Model):
    """Variational Autoencoder (VAE) model composed of an encoder and a decoder."""

    def __init__(self, input_shape, latent_dim=2, activation='relu'):
        """Initialize the VAE with given parameters."""
        super(VariationalAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(input_shape, latent_dim, activation)
        self.decoder = self.build_decoder(input_shape, latent_dim, activation)

    def build_encoder(self, input_shape, latent_dim, activation):
        """Build the encoder model that maps input to latent space."""
        inputs = Input(shape=input_shape)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)
        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self, input_shape, latent_dim, activation):
        """Build the decoder model that maps latent space back to the input space."""
        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x = layers.Dense(7 * 7 * 128, activation=activation)(latent_inputs)
        x = layers.Reshape((7, 7, 128))(x)
        x = layers.Conv2DTranspose(128, 3, activation=activation, strides=2, padding='same')(x)
        x = layers.Conv2DTranspose(64, 3, activation=activation, strides=2, padding='same')(x)
        x = layers.Conv2DTranspose(32, 3, activation=activation, strides=1, padding='same')(x)
        outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)  # Ajustement ici
        return tf.keras.Model(latent_inputs, outputs, name='decoder')

    def call(self, inputs):
        """Forward pass through the VAE."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_sum(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1, axis=-1)
        self.add_loss(kl_loss)
        return reconstructed

    def compile(self, optimizer, loss):
        """Compile the VAE with the given optimizer and loss function."""
        super(VariationalAutoEncoder, self).compile(optimizer=optimizer)
        self.compiled_loss = loss
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, data):
        """One step of training."""
        with tf.GradientTape() as tape:
            reconstructed = self(data)
            reconstruction_loss = tf.reduce_mean(self.compiled_loss(data, reconstructed))
            total_loss = reconstruction_loss + sum(self.losses)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        """One step of validation/testing."""
        reconstructed = self(data)
        reconstruction_loss = tf.reduce_mean(self.compiled_loss(data, reconstructed))
        total_loss = reconstruction_loss + sum(self.losses)
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def encode(self, data):
        """Encode the input data into the latent space."""
        _, _, z = self.encoder.predict(data)
        return z

    def decode(self, latent_data):
        """Decode the latent space data back to input space."""
        return self.decoder.predict(latent_data)

    def fit_model(self, x_train, x_test, epochs, batch_size):
        """Train the VAE model."""
        return self.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))

    def generate_synthetic_data(self, num_samples=16):
        """Generate synthetic data by sampling the latent space."""
        random_latent_vectors = np.random.normal(size=(num_samples, self.latent_dim))
        generated_images = self.decoder.predict(random_latent_vectors)
        return generated_images, random_latent_vectors
