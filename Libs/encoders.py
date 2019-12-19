from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import numpy as np
from enum import Enum

def get_mnist_encoder():
    print("=== Preparing MNIST encoder ===")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
    x_train_mnist = x_train_mnist.reshape(x_train_mnist.shape[0], 28, 28, 1)
    x_train_mnist = x_train_mnist.astype('float32')
    x_train_mnist /= 255

    encoding_dim = 20
    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    x_train_mnist_reshaped = x_train_mnist.reshape((len(x_train_mnist), np.prod(x_train_mnist.shape[1:])))
    x_test_mnist_reshaped = x_test_mnist.reshape((len(x_test_mnist), np.prod(x_test_mnist.shape[1:])))
    autoencoder.fit(x_train_mnist_reshaped, x_train_mnist_reshaped,
                    epochs=50,
                    batch_size=256,
                    #shuffle=True,
                    validation_data=(x_test_mnist_reshaped, x_test_mnist_reshaped), verbose=0)
    autoencoder.layers.pop()
    return autoencoder


def get_mnist_encoder_2():
    encoder = get_mnist_encoder()
    return Model(encoder.input, encoder.layers[-1].output)


def get_fmnist_encoder():
    print("=== Preparing FMNIST encoder ===")
    (x_train_fmnist, y_train_fmnist), (x_test_fmnist, y_test_fmnist) = tf.keras.datasets.fashion_mnist.load_data()
    x_train_fmnist = x_train_fmnist.reshape(x_train_fmnist.shape[0], 28, 28, 1)
    x_train_fmnist = x_train_fmnist.astype('float32')
    x_train_fmnist /= 255

    encoding_dim = 32
    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    decoded = Dense(784, activation='sigmoid')(encoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    x_train_fmnist_reshaped = x_train_fmnist.reshape((len(x_train_fmnist), np.prod(x_train_fmnist.shape[1:])))
    x_test_fmnist_reshaped = x_test_fmnist.reshape((len(x_test_fmnist), np.prod(x_test_fmnist.shape[1:])))
    autoencoder.fit(x_train_fmnist_reshaped, x_train_fmnist_reshaped,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test_fmnist_reshaped, x_test_fmnist_reshaped), verbose=0)
    autoencoder.layers.pop()
    return autoencoder


def get_fmnist_encoder_2():
    encoder = get_fmnist_encoder()
    return Model(encoder.input, encoder.layers[-1].output)


class Encoders(Enum):
    mnist_1 = get_mnist_encoder
    mnist_2 = get_mnist_encoder_2
    fmnist_1 = get_fmnist_encoder
    fmnist_2 = get_fmnist_encoder_2

    def get_by_name(name):
        if name == 'mnist_1':
            return Encoders.mnist_1
        if name == 'mnist_2':
            return Encoders.mnist_2
        if name == 'fmnist_1':
            return Encoders.fmnist_1
        if name == 'fmnist_2':
            return Encoders.fmnist_2
        return None










