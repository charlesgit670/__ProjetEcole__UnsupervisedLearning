import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from load_image import load_data_food  # Import de la fonction load_data_food depuis le fichier load_image.py
from PIL import Image


def load_mnist_data(normalize=True, shuffle=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    if normalize:
        x_train /= 255.
        x_test /= 255.

    if shuffle:
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return x_train, x_test, y_train, y_test


def load_food_data(target_size=64):
    X_train, X_test, y_train, y_test, label_to_classname = load_data_food()

    # Redimensionner les images à la taille cible
    X_train_resized = np.array([np.array(Image.fromarray(img).resize((target_size, target_size))) for img in X_train])
    X_test_resized = np.array([np.array(Image.fromarray(img).resize((target_size, target_size))) for img in X_test])

    X_train_resized = X_train_resized.reshape((len(X_train_resized), -1)).astype('float32') / 255.
    X_test_resized = X_test_resized.reshape((len(X_test_resized), -1)).astype('float32') / 255.

    return X_train_resized, X_test_resized, y_train, y_test, label_to_classname
