from utils import cut_ds
import init_generators as igen

from keras.callbacks import EarlyStopping
import numpy as np


def run_AL(
        query_f,
        create_model,
        reshape_f,
        score_f,
        x_train, y_train, x_test, y_test,
        init_size, batch_size, total_size, query_number, random_state
):
    x_train, y_train = cut_ds(total_size, x_train, y_train, random_state)

    labeled, x_train_labeled, y_train_labeled, model = default_init(
        x_train, y_train, x_test, y_test, init_size, create_model
    )

    return default_learner(x_train, y_train,
                           labeled, x_train_labeled, y_train_labeled,
                           x_test, y_test,
                           score_f, query_number, query_f, model, batch_size, reshape_f
                           )


def run_AL_SBC(
        query_f,
        create_model,
        reshape_f,
        score_f,
        x_train, y_train, x_test, y_test,
        init_size, batch_size, total_size, query_number, random_state,
        encoder
):
    x_train, y_train = cut_ds(total_size, x_train, y_train, random_state)

    labeled, x_train_labeled, y_train_labeled, model = sbc_init(
        x_train, y_train, x_test, y_test, init_size, encoder, create_model
    )

    return default_learner(x_train, y_train,
                           labeled, x_train_labeled, y_train_labeled,
                           x_test, y_test,
                           score_f, query_number, query_f, model, batch_size, reshape_f
                           )


def run_AL_SUD(
        query_f,
        create_model,
        reshape_f,
        score_f,
        x_train, y_train, x_test, y_test,
        init_size, batch_size, total_size, query_number, random_state,
        encoder
):
    x_train, y_train = cut_ds(total_size, x_train, y_train, random_state)

    labeled, x_train_labeled, y_train_labeled, model = default_init(
        x_train, y_train, x_test, y_test, init_size, create_model
    )

    return sud_learner(x_train, y_train,
                       labeled, x_train_labeled, y_train_labeled,
                       x_test, y_test,
                       encoder, init_size, score_f, query_number, query_f, model, batch_size, reshape_f
                       )


def run_AL_SBC_SUD(
        query_f,
        create_model,
        reshape_f,
        score_f,
        x_train, y_train, x_test, y_test,
        init_size, batch_size, total_size, query_number, random_state,
        encoder
):
    x_train, y_train = cut_ds(total_size, x_train, y_train, random_state)

    labeled, x_train_labeled, y_train_labeled, model = sbc_init(
        x_train, y_train, x_test, y_test, init_size, encoder, create_model
    )

    return sud_learner(x_train, y_train,
                       labeled, x_train_labeled, y_train_labeled,
                       x_test, y_test,
                       encoder, init_size, score_f, query_number, query_f, model, batch_size, reshape_f
                       )


def sbc_init(x_train, y_train, x_test, y_test, init_size, encoder, create_model):
    labeled, x_train_labeled, y_train_labeled = igen.cluster_init(x_train, y_train, init_size, encoder)
    model = create_model()
    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)
    model.fit(x_train_labeled, y_train_labeled, validation_data=(x_test, y_test), epochs=20, callbacks=[es], verbose=0)
    return labeled, x_train_labeled, y_train_labeled, model


def default_init(x_train, y_train, x_test, y_test, init_size, create_model):
    labeled, x_train_labeled, y_train_labeled = igen.default_init(x_train, y_train, init_size)
    model = create_model()
    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)
    model.fit(x_train_labeled, y_train_labeled, validation_data=(x_test, y_test), epochs=20, callbacks=[es], verbose=0)
    return labeled, x_train_labeled, y_train_labeled, model


def sud_learner(x_train, y_train,
                labeled, x_train_labeled, y_train_labeled,
                x_test, y_test,
                encoder, init_size, score_f, query_number, query_f, model, batch_size, reshape_f
                ):
    x_reshaped = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_encoded = encoder.predict(x_reshaped)
    x_encoded = [e.flatten() for e in x_encoded]

    center_vector = np.sum([x_encoded[i].flatten() for i in range(len(x_train)) if not labeled[i]], axis=0)
    unlabeled_size = len(x_train) - init_size
    mass_center = center_vector / unlabeled_size

    acc_growth = [(x_train_labeled.shape[0], score_f(model, x_test, y_test))]
    for i in range(query_number):
        query = query_f(model, x_train, x_encoded, labeled, batch_size, reshape_f, mass_center)
        for i in query:
            x_train_labeled = np.append(x_train_labeled, [x_train[i]], axis=0)
            y_train_labeled = np.append(y_train_labeled, [y_train[i]], axis=0)
            labeled[i] = True
            center_vector -= x_encoded[i]
        unlabeled_size -= batch_size
        mass_center = center_vector / unlabeled_size
        model.fit(x_train_labeled, y_train_labeled, validation_data=(x_test, y_test), verbose=0)
        acc_growth += [(x_train_labeled.shape[0], score_f(model, x_test, y_test))]
    return acc_growth


def default_learner(x_train, y_train,
                    labeled, x_train_labeled, y_train_labeled,
                    x_test, y_test,
                    score_f, query_number, query_f, model, batch_size, reshape_f
                    ):
    acc_growth = [(x_train_labeled.shape[0], score_f(model, x_test, y_test))]
    for i in range(query_number):
        query = query_f(model, x_train, labeled, batch_size, reshape_f)
        for i in query:
            x_train_labeled = np.append(x_train_labeled, [x_train[i]], axis=0)
            y_train_labeled = np.append(y_train_labeled, [y_train[i]], axis=0)
            labeled[i] = True
        model.fit(x_train_labeled, y_train_labeled, validation_data=(x_test, y_test), verbose=0)
        acc_growth += [(x_train_labeled.shape[0], score_f(model, x_test, y_test))]
    return acc_growth
