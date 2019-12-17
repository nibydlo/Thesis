import learners
import plots
import datasets
import encoders
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import seed
import init_generators
import random as rn
import tensorflow.python.keras.backend as K
import keras
INIT_SIZE = 2000
TRAIN_SIZE = 10000
QUERIED_SIZE = 300

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


mnist_input_shape = (28, 28, 1)
es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)


def create_sequential_model():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=mnist_input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model


np.random.seed(42)
rn.seed(12345)
tf.compat.v1.set_random_seed(1234)
x_train, y_train, x_test, y_test = datasets.get_mnist()
#is_labeled, x_labeled, y_labeled = init_generators.default_init(x_train, y_train, INIT_SIZE)
x_train1, y_train1 = x_train.copy(), y_train.copy()
x_train2, y_train2 = x_train.copy(), y_train.copy()

tf.compat.v1.keras.backend.clear_session()

np.random.seed(42)
rn.seed(12345)
tf.compat.v1.set_random_seed(1234)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
print("random:", rn.random())
print("numpy:", np.random.random(1))
print("tf:", tf.random.uniform((2, 2)))
keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=239)
model1 = create_sequential_model()
print(model1.evaluate(x_test, y_test, verbose=0))
model1.fit(x_train[:6000], y_train[:6000], 25, validation_data=(x_test, y_test), shuffle=False, epochs=1, verbose=0)
print(model1.evaluate(x_test, y_test, verbose=0))

tf.compat.v1.keras.backend.clear_session()


np.random.seed(42)
rn.seed(12345)
tf.compat.v1.set_random_seed(1234)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
print("random:", rn.random())
print("numpy:", np.random.random(1))
print("tf:", tf.random.uniform((2, 2)))
keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=239)
model2 = create_sequential_model()
print(model2.evaluate(x_test, y_test, verbose=0))
model2.fit(x_train[:6000], y_train[:6000], 25, validation_data=(x_test, y_test), shuffle=False, epochs=1, verbose=0)
print(model2.evaluate(x_test, y_test, verbose=0))

plt.legend()
plt.show()

