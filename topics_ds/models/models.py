import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Multiply, Add, Lambda, Reshape, Flatten, dot
from highway import Highway
from tensorflow.keras import regularizers

from keras_radam.training import RAdamOptimizer

def get_model_trivial():
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x = concatenate([inp_img, inp_txt])
    x = Dense(64)(x)
    x = Dropout(0.5)(x)
    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_default_lr():
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(64, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(64, activation='relu')(x_img)

    x_txt = Dense(64, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(64, activation='relu')(x_txt)

    x = concatenate([x_img, x_txt])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_custom_lr(custom_lr=1e-3):
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(64, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(64, activation='relu')(x_img)

    x_txt = Dense(64, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(64, activation='relu')(x_txt)

    x = concatenate([x_img, x_txt])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=custom_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_default_lr_wide():
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(512, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(256, activation='relu')(x_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(256, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(256, activation='relu')(x_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_custom_lr_wide(custom_lr=1e-3):
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(512, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(256, activation='relu')(x_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(256, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(256, activation='relu')(x_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=custom_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_no_img_flag(custom_lr=1e-3):
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))
    inp_img_flag = Input(shape=(1,))

    x_img = Dense(512, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(256, activation='relu')(x_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(256, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(256, activation='relu')(x_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt, inp_img_flag])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt, inp_img_flag], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=custom_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_no_txt_flag(custom_lr=1e-3):
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))
    inp_txt_flag = Input(shape=(1,))

    x_img = Dense(512, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(256, activation='relu')(x_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(256, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(256, activation='relu')(x_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt, inp_txt_flag])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt, inp_txt_flag], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=custom_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_no_img_txt_flag(custom_lr=1e-3):
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))
    inp_img_flag = Input(shape=(1,))
    inp_txt_flag = Input(shape=(1,))

    x_img = Dense(512, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(256, activation='relu')(x_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(256, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(256, activation='relu')(x_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt, inp_img_flag, inp_txt_flag])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt, inp_img_flag, inp_txt_flag], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=custom_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_custom_lr_wide_reg(custom_lr=1e-3, regularizer=regularizers.l1(0.001)):
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(
        512,
        activation='relu',
        kernel_regularizer=regularizer
    )(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizer
    )(x_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizer
    )(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizer
    )(x_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt])
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizer
    )(x)
    x = Dropout(0.25)(x)
    x = Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizer
    )(x)
    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=custom_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_model_mix():
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(1024, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(512, activation='relu')(x_img)

    x_txt = Dense(512, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(512, activation='relu')(x_txt)

    add = Add()([x_img, x_txt])
    add = Dense(512, activation='relu')(add)
    add = Dropout(0.25)(add)

    mult = Multiply()([x_img, x_txt])
    mult = Dense(512, activation='relu')(mult)
    mult = Dropout(0.25)(mult)

    mix = concatenate([add, mult])
    mix = Dense(512, activation='relu')(mix)
    mix = Dropout(0.25)(mix)

    out = Dense(50, activation='softmax')(mix)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    return model


def get_model_mix():
    model = create_model_mix()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_mix_custom_lr(custom_lr=1e-3):
    model = create_model_mix()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=custom_lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_2_hw():
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x = concatenate([inp_img, inp_txt])
    x = Highway(activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Highway(activation='relu')(x)
    x = Dropout(0.25)(x)

    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_2_hw_reg(W_regularizer=regularizers.l1(0.01), b_regularizer=regularizers.l1(0.01)):
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x = concatenate([inp_img, inp_txt])
    x = Highway(activation='relu', W_regularizer=W_regularizer, b_regularizer=b_regularizer)(x)
    x = Dropout(0.25)(x)
    x = Highway(activation='relu', W_regularizer=W_regularizer, b_regularizer=b_regularizer)(x)
    x = Dropout(0.25)(x)

    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_matrix_fusion_short(matrix_size=64):
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(matrix_size, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Reshape(target_shape=(1,matrix_size))(x_img)

    x_txt = Dense(matrix_size, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Reshape(target_shape=(1,matrix_size))(x_txt)

    con = concatenate([x_img, x_txt], axis=-2)
    x = Lambda(lambda a: tf.map_fn(lambda e: tf.tensordot(e[0], e[1], axes=0), a))(con)
    row_pooling = Lambda(lambda arg: tf.math.reduce_max(arg, axis=2))(x)
    col_pooling = Lambda(lambda arg: tf.math.reduce_max(arg, axis=1))(x)
    x = concatenate([row_pooling, col_pooling])

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)

    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_matrix_fusion(matrix_size=64):
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(512, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Dense(matrix_size, activation='relu')(x_img)
    x_img = Dropout(0.25)(x_img)
    x_img = Reshape(target_shape=(1,matrix_size))(x_img)

    x_txt = Dense(256, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Dense(matrix_size, activation='relu')(x_txt)
    x_txt = Dropout(0.25)(x_txt)
    x_txt = Reshape(target_shape=(1,matrix_size))(x_txt)

    con = concatenate([x_img, x_txt], axis=-2)
    x = Lambda(lambda a: tf.map_fn(lambda e: tf.tensordot(e[0], e[1], axes=0), a))(con)
    row_pooling = Lambda(lambda arg: tf.math.reduce_max(arg, axis=2))(x)
    col_pooling = Lambda(lambda arg: tf.math.reduce_max(arg, axis=1))(x)
    x = concatenate([row_pooling, col_pooling])

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)

    out = Dense(50, activation='softmax')(x)

    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_residual_concat():
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(128, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(128, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = concatenate([x, x_img, x_txt])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)

    out = Dense(50, activation='softmax')(x)
    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    return model


def get_model_residual_concat():
    model = create_model_residual_concat()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_residual_concat_radam():
    model = create_model_residual_concat()
    optimizer = RAdamOptimizer(total_steps=5000, warmup_proportion=0.1, min_lr=1e-5)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_residual_mult():
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(128, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(128, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Multiply()([x, concatenate([x_img, x_txt])])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)

    out = Dense(50, activation='softmax')(x)
    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_residual_none():
    inp_img = Input(shape=(1024,))
    inp_txt = Input(shape=(300,))

    x_img = Dense(128, activation='relu')(inp_img)
    x_img = Dropout(0.25)(x_img)

    x_txt = Dense(128, activation='relu')(inp_txt)
    x_txt = Dropout(0.25)(x_txt)

    x = concatenate([x_img, x_txt])
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)

    out = Dense(50, activation='softmax')(x)
    model = Model(inputs=[inp_img, inp_txt], outputs=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model