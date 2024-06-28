import numpy as np
import tensorflow as tf
from keras import layers
import keras

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(latent_dim, input_shape):
    """Build the encoder model."""
    encoder_inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def build_decoder(latent_dim, input_shape):
    """Build the decoder model."""
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense((input_shape[0] // 4) * (input_shape[1] // 4) * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((input_shape[0] // 4, input_shape[1] // 4, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(input_shape[2], 3, activation="sigmoid", padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


class VAE(keras.Model):
    """Variational Autoencoder (VAE) model composed of an encoder and a decoder."""

    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    @tf.function
    def train_step(self, data):
        assert data is not None, "Input data cannot be None"
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            tf.debugging.assert_all_finite(z_mean, "z_mean contains NaN values")
            tf.debugging.assert_all_finite(z_log_var, "z_log_var contains NaN values")
            tf.debugging.assert_all_finite(z, "z contains NaN values")

            reconstruction = self.decoder(z)
            tf.debugging.assert_all_finite(reconstruction, "reconstruction contains NaN values")

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + 0.1 * kl_loss

            tf.debugging.assert_all_finite(total_loss, "total_loss contains NaN values")
        grads = tape.gradient(total_loss, self.trainable_weights)
        for grad in grads:
            if grad is not None:
                tf.debugging.assert_all_finite(grad, "Gradients contain NaN values")
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def generate_synthetic_data(self, latent_dim, step=1):
        """Generate synthetic data by sampling from the latent space."""

        if latent_dim == 2:
            ranges = np.arange(-7, 4 + step, step)
            latent_points = np.array([[x, y] for x in ranges for y in ranges])
        elif latent_dim == 3:
            step=1
            ranges = np.arange(-5, 4 + step, step)
            latent_points = np.array([[x, y, z] for x in ranges for y in ranges for z in ranges])
        else:
            latent_points = np.random.normal(size=(100, latent_dim))

        generated_images = self.decoder(latent_points)
        return generated_images, latent_points

def create_vae(latent_dim, input_shape):
    encoder = build_encoder(latent_dim, input_shape)
    decoder = build_decoder(latent_dim, input_shape)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    return vae


def create_vae(latent_dim, input_shape):
    encoder = build_encoder(latent_dim, input_shape)
    decoder = build_decoder(latent_dim, input_shape)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanSquaredError())
    return vae
