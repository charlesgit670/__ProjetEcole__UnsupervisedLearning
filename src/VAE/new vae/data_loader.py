import numpy as np
from tensorflow.keras.datasets import mnist

def load_mnist(normalize=True):
    """Load MNIST dataset and normalize it."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1).astype("float32")
    x_test = np.expand_dims(x_test, -1).astype("float32")
    if normalize:
        x_train /= 255.0
        x_test /= 255.0
    return (x_train, y_train), (x_test, y_test)
