import numpy as np
import tensorflow as tf

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
