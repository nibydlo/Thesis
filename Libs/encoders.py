from keras.layers import Input, Dense
from keras.models import Model


def get_mnist_encoder():
    encoding_dim = 32
    input_img = Input(shape=(784,))
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    encoder = Model(input_img, encoded)
    return encoder

