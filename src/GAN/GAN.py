import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import tqdm

import matplotlib.pyplot as plt
from IPython import display




def gan(random_normal_dimensions, lr=0.001):
    generator = keras.models.Sequential([
        keras.layers.Dense(64, activation='selu', input_shape=[random_normal_dimensions]),
        # keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(128, activation='selu'),
        # keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(28*28, activation='sigmoid'),
        keras.layers.Reshape([28, 28])
    ])

    discriminator = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(128, activation='selu'),
        # keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(64, activation='selu'),
        # keras.layers.LeakyReLU(0.2),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr))
    gan = keras.models.Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr))

    return generator, discriminator, gan


def create_mnist_dataset(batch_size):
    (X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()
    X = np.concat([X_train, X_test])
    X = X / 255.0
    size = len(X)
    number_of_batch = size // batch_size
    X = X[:number_of_batch*batch_size]
    dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(len(X)).batch(batch_size)
    return dataset


def plot_results(images, n_cols=None):
    '''visualizes fake images'''
    display.clear_output(wait=False)

    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1

    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    plt.figure(figsize=(n_cols, n_rows))

    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")

def plot_loss(discriminator_losses, generator_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.plot(generator_losses, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


BATCH_SIZE = 64
REAL_BATCH_SIZE = BATCH_SIZE // 2
FAKE_BATCH_SIZE = BATCH_SIZE // 2
EPOCHS = 20
RANDOM_NORMAL_DIMENSIONS = 32
N = 5

dataset = create_mnist_dataset(REAL_BATCH_SIZE)
generator, discriminator, gan = gan(RANDOM_NORMAL_DIMENSIONS, 0.001)

discriminator_losses = []
generator_losses = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    d_loss_epoch = 0
    g_loss_epoch = 0
    batch_count = 0

    for real_images in tqdm(dataset):
        noise = tf.random.normal(shape=[FAKE_BATCH_SIZE, RANDOM_NORMAL_DIMENSIONS])
        fake_images = generator(noise)
        mixed_images = np.concat([fake_images, real_images], axis=0)
        discriminator_labels = np.array([0]*FAKE_BATCH_SIZE + [1]*REAL_BATCH_SIZE)

        # Train discriminator
        discriminator.trainable = True
        for _ in range(N):
            d_loss = discriminator.train_on_batch(mixed_images, discriminator_labels)
        d_loss_epoch += d_loss

        # Train generator
        discriminator.trainable = False
        noise = tf.random.normal(shape=[BATCH_SIZE, RANDOM_NORMAL_DIMENSIONS])

        generator_labels = np.array([1]*BATCH_SIZE)
        g_loss = gan.train_on_batch(noise, generator_labels)
        g_loss_epoch += g_loss

        batch_count += 1

    discriminator_losses.append(d_loss_epoch / batch_count)
    generator_losses.append(g_loss_epoch / batch_count)

    print(f"Discriminator loss : {d_loss_epoch / batch_count}")
    print(f"Generator loss : {g_loss_epoch / batch_count}")

plot_results(fake_images, 8)
plt.show()

plot_loss(discriminator_losses, generator_losses)
