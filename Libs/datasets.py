import tensorflow as tf


def get_mnist():
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
    x_train_mnist, x_test_mnist = reshape_mnist(x_train_mnist, x_test_mnist)
    return x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist


def get_fmnist():
    (x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = tf.keras.datasets.fashion_mnist.load_data()
    x_train_fmnist, x_test_fmnist = reshape_mnist(x_train_fmnist, x_test_fmnist)
    return x_train_fmnist, y_train_fmnist, x_test_fmnist, y_test_fmnist


def reshape_mnist(x_train, x_test):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, x_test
