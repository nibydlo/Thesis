from utils import cut_ds
import init_generators as igen

from keras.callbacks import EarlyStopping
from enum import Enum
import numpy as np
import progressbar
import copy

from queries import Queries
from scores import Scores
from models import Models
from encoders import Encoders

class Config:
    """
    config : {
        score - function from Scores enum to get model score on each step
        query - function from Query enum to ask for labeling data
        model - function from Models enum to create one of described models
        encoder - function from Encoders to create encoder
        init_size - size of initial labeled data_set
        batch_size - how many objects are asked to be labeled
        total_size - part of the dataset that is used in experiment
        queries_number - number of queries to accessor
        random_state - used while cutting the train dataset
    }
    """

    def __init__(self, config):
        self.query_f = Queries.get_by_name(config['query'])
        self.score_f = Scores.get_by_name(config['score'])
        self.create_model = Models.get_by_name(config['model'])
        if 'encoder' in config:
            self.create_encoder = Encoders.get_by_name(config['encoder'])
        self.init_size = config['init_size']
        self.batch_size = config['batch_size']
        self.total_size = config['total_size']
        self.queries_number = config['queries_number']
        self.random_state = config['random_state']

    def set_query_f(self, raw_query):
        query = Queries.get_by_name(raw_query)
        res = copy.deepcopy(self)
        res.query_f = query
        return res

    def set_model(self, raw_model):
        model = Models.get_by_name(raw_model)
        res = copy.deepcopy(self)
        res.create_model = model
        return res

    def set_encoder(self, raw_encoder):
        encoder = Encoders.get_by_name(raw_encoder)
        res = copy.deepcopy(self)
        res.create_encoder = encoder
        return res

    def set_score_f(self, raw_score):
        score = Scores.get_by_name(raw_score)
        res = copy.deepcopy(self)
        res.score_f = score
        return res

    def set_init_size(self, init_size):
        res = copy.deepcopy(self)
        res.init_size = init_size
        return res

    def set_batch_size(self, batch_size):
        res = copy.deepcopy(self)
        res.batch_size = batch_size
        return res

    def set_total_size(self, total_size):
        res = copy.deepcopy(self)
        res.total_size = total_size
        return res

    def set_queries_number(self, queries_number):
        res = copy.deepcopy(self)
        res.queries_number = queries_number
        return res

    def set_random_state(self, random_state):
        res = copy.deepcopy(self)
        res.random_state = random_state
        return res


def run_AL(
        query_f,
        create_model,
        score_f,
        x, y, val_data,
        init_size, batch_size, total_size, queries_number, random_state
):
    x, y = cut_ds(total_size, x, y, random_state)

    labeled, x_labeled, y_labeled, model = default_init(
        x, y, val_data, init_size, create_model
    )

    return default_learner(x, y, labeled, x_labeled, y_labeled, val_data,
                           score_f, queries_number, query_f, model, batch_size
                           )


def run_AL_SBC(
        query_f,
        create_model,
        score_f,
        x, y, val_data,
        init_size, batch_size, total_size, queries_number, random_state,
        create_encoder
):
    encoder = create_encoder()
    x, y = cut_ds(total_size, x, y, random_state)

    labeled, x_labeled, y_labeled, model = sbc_init(
        x, y, val_data, init_size, encoder, create_model
    )

    return default_learner(x, y, labeled, x_labeled, y_labeled, val_data,
                           score_f, queries_number, query_f, model, batch_size
                           )


def run_AL_SUD(
        query_f,
        create_model,
        score_f,
        x, y, val_data,
        init_size, batch_size, total_size, queries_number, random_state,
        create_encoder, k, b
):
    encoder = create_encoder()
    print("=== SUD scenario started ===")
    x, y = cut_ds(total_size, x, y, random_state)

    labeled, x_labeled, y_labeled, model = default_init(
        x, y, val_data, init_size, create_model
    )

    return sud_learner(x, y,
                       labeled, x_labeled, y_labeled,
                       val_data,
                       encoder, score_f, queries_number, query_f, model, batch_size, k, b
                       )


def run_AL_SBC_SUD(
        query_f,
        create_model,
        score_f,
        x, y, val_data,
        init_size, batch_size, total_size, queries_number, random_state,
        create_encoder, k, b
):
    encoder = create_encoder()
    x, y = cut_ds(total_size, x, y, random_state)

    labeled, x_labeled, y_labeled, model = sbc_init(
        x, y, val_data, init_size, encoder, create_model
    )

    return sud_learner(x, y, labeled, x_labeled, y_labeled, val_data,
                       encoder, score_f, queries_number, query_f, model, batch_size, k, b
                       )


def init_model(x_labeled, y_labeled, val_data, create_model):
    print("=== Start model initialization ===")
    model = create_model()
    es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)
    model.fit(x_labeled, y_labeled, validation_data=val_data, epochs=20, callbacks=[es], verbose=0)
    print("=== Finish model initialization ===")
    return model


def sbc_init(x, y, val_data, init_size, encoder, create_model):
    print("=== Start data clustering ===")
    labeled, x_labeled, y_labeled = igen.cluster_init(x, y, init_size, encoder)
    print("=== Finish data clustering ===")
    model = init_model(x_labeled, y_labeled, val_data=val_data, create_model=create_model)
    return labeled, x_labeled, y_labeled, model


def default_init(x, y, val_data, init_size, create_model):
    labeled, x_labeled, y_labeled = igen.default_init(x, y, init_size)
    model = init_model(x_labeled, y_labeled, val_data=val_data, create_model=create_model)
    return labeled, x_labeled, y_labeled, model


def sud_learner(x, y, labeled, x_labeled, y_labeled, val_data,
                encoder, score_f, query_number, query_f, model, batch_size, k, b
                ):
    x_reshaped = x.reshape((len(x), np.prod(x.shape[1:])))
    x_encoded = encoder.predict(x_reshaped)
    x_encoded = [e.flatten() for e in x_encoded]

    # center_vector = np.sum([x_encoded[i].flatten() for i in range(len(x_train)) if not labeled[i]], axis=0)
    # unlabeled_size = len(x_train) - init_size
    # mass_center = center_vector / unlabeled_size

    acc_growth = [(x_labeled.shape[0], score_f(model, val_data))]
    print("=== SUD learning started === ")
    with progressbar.ProgressBar(maxval=query_number, redirect_stdout=True) as bar:
        for i in range(query_number):
            query = query_f(
                model=model,
                x_train=x,
                x_encoded=x_encoded,
                labeled=labeled,
                batch_size=batch_size,
                k=k,
                b=b
            )
            for j in query:
                x_labeled = np.append(x_labeled, [x[j]], axis=0)
                y_labeled = np.append(y_labeled, [y[j]], axis=0)
                labeled[j] = True
                # center_vector -= x_encoded[j]
            # unlabeled_size -= batch_size
            # mass_center = center_vector / unlabeled_size
            model.fit(x_labeled, y_labeled, validation_data=val_data, verbose=0)
            acc_growth += [(x_labeled.shape[0], score_f(model, val_data))]
            bar.update(i)
    print("=== SUD learning finished ===")
    return acc_growth


def default_learner(x, y, labeled, x_labeled, y_labeled, val_data,
                    score_f, query_number, query_f, model, batch_size
                    ):
    acc_growth = [(x_labeled.shape[0], score_f(model, val_data))]
    print("=== uncertainty only learning started === ")
    with progressbar.ProgressBar(maxval=query_number, redirect_stdout=True) as bar:
        for i in range(query_number):
            query = query_f(model, x, labeled, batch_size)
            for j in query:
                x_labeled = np.append(x_labeled, [x[j]], axis=0)
                y_labeled = np.append(y_labeled, [y[j]], axis=0)
                labeled[j] = True
            model.fit(x_labeled, y_labeled, validation_data=val_data, verbose=0)
            acc_growth += [(x_labeled.shape[0], score_f(model, val_data))]
            bar.update(i)
    print("=== uncertainty only learning finished ===")
    return acc_growth


class Scenarios(Enum):
    uncertainty = run_AL
    sbc = run_AL_SBC
    sud = run_AL_SUD
    sbc_sud = run_AL_SBC_SUD

print(type(Scenarios.uncertainty))

class Experiment:

    def __init__(self, config, raw_scenario, x, y, val_data):
        self.config = config
        self.raw_scenario = raw_scenario
        self.x = x
        self.y = y
        self.val_data = val_data

    def run(self):
        res = None
        if self.raw_scenario == 'uncertainty':
            res = run_AL(
                query_f=self.config.query_f,
                create_model=self.config.create_model,
                score_f=self.config.score_f,
                init_size=self.config.init_size,
                batch_size=self.config.batch_size,
                total_size=self.config.total_size,
                queries_number=self.config.queries_number,
                random_state=self.config.random_state,
                x=self.x,
                y=self.y,
                val_data=self.val_data
            )
        if self.raw_scenario == 'sbc':
            res = run_AL_SBC(
                query_f=self.config.query_f,
                create_model=self.config.create_model,
                score_f=self.config.score_f,
                init_size=self.config.init_size,
                batch_size=self.config.batch_size,
                total_size=self.config.total_size,
                queries_number=self.config.queries_number,
                random_state=self.config.random_state,
                x=self.x,
                y=self.y,
                val_data=self.val_data,
                create_encoder=self.config.create_encoder
            )
        return res

