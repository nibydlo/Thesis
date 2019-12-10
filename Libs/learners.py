import numpy as np
from sklearn.utils import shuffle
import random
from keras.callbacks import EarlyStopping
import progressbar

from queries import query_entropy, query_entropy_sud
import init_generators


def cut_ds(short_size, x, y, random_s):
    x_shuffled, y_shuffled = shuffle(x, y, random_state=random_s)
    x_shuffled = x_shuffled[:short_size]
    y_shuffled = y_shuffled[:short_size]
    return x_shuffled, y_shuffled


class AbstractLearner:
    def __init__(self):
        self.acc_growth = []

    def compile(
            self,
            config=None,
            init_size=None,
            train_size=None,
            queries_number=None,
            batch_size=1,
            random_state=random.randint(0, 20)
    ):
        if config is not None:
            if 'init_size' in config:
                self.init_size = config['init_size']
            if 'train_size' in config:
                self.train_size = config['train_size']
            if 'queries_number' in config:
                self.queries_number = config['queries_number']
            if 'batch_size' in config:
                self.batch_size = config['batch_size']
            if 'random_state' in config:
                self.random_state = config['random_state']
        else:
            self.init_size = init_size
            self.train_size = train_size
            self.queries_number = queries_number
            self.batch_size = batch_size
            self.random_state = random_state

    def get_init_size(self):
        return self.init_size

    def get_train_size(self):
        return self.train_size

    def get_queries_number(self):
        return self.queries_number

    def get_batch_size(self):
        return self.batch_size

    def get_random_state(self):
        return self.random_state

    def get_stat(self):
        return self.stat

    def put_stat(self, stat):
        self.stat = stat

    def put_query_process(self, query_process):
        self.query_process = query_process

    def put_get_labeled(self, f):
        self.get_labeled_set = f

    def put_encoder(self, encoder):
        self.encoder = encoder

    def get_labeled_set(self, init_size, x, y):
        print("=== preparing standard labeled data ===")
        return init_generators.default_init(x, y, init_size)

    def query_process(self, x, y, labeled, x_labeled, y_labeled, val_data, model):
        self.stat = []
        if val_data is not None:
            self.stat.append((x_labeled.shape[0], model.evaluate(val_data[0], val_data[1], verbose=0)[1]))
        print("=== uncertainty only learning started === ")
        with progressbar.ProgressBar(maxval=self.queries_number, redirect_stdout=True) as bar:
            for i in range(self.queries_number):
                query = query_entropy(model, x, labeled, self.batch_size)
                for j in query:
                    x_labeled = np.append(x_labeled, [x[j]], axis=0)
                    y_labeled = np.append(y_labeled, [y[j]], axis=0)
                    labeled[j] = True
                model.fit(x_labeled, y_labeled, validation_data=val_data, verbose=0)
                if val_data is not None:
                    self.stat.append((x_labeled.shape[0], model.evaluate(val_data[0], val_data[1], verbose=0)[1]))
                bar.update(i)
        print("=== uncertainty only learning finished ===")


class EntropyLearner(AbstractLearner):

    def learn(self, model, x, y, validation_data=None):
        init_size = self.init_size if self.init_size is not None and self.init_size <= len(x) else len(x)
        train_size = self.train_size if self.train_size is not None and self.train_size <= len(x) else len(x)
        self.queries_number = self.queries_number \
            if self.queries_number is not None and self.queries_number <= train_size - init_size \
            else train_size - init_size

        x, y = cut_ds(train_size, x, y, self.random_state)
        is_labeled, x_labeled, y_labeled = self.get_labeled_set(init_size, x, y)

        print("=== preparing initial model ===")
        es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=3)
        model.fit(x_labeled, y_labeled, validation_data=validation_data, epochs=20, callbacks=[es], verbose=0)
        self.query_process(x, y, is_labeled, x_labeled, y_labeled, validation_data, model)


class AbstractLearnerDecorator(AbstractLearner):

    def __init__(self, decorated_learner):
        super().__init__()
        self.decorated_learner = decorated_learner

        self.learn = decorated_learner.learn
        self.compile = decorated_learner.compile
        self.get_stat = decorated_learner.get_stat
        self.put_stat = decorated_learner.put_stat
        self.put_query_process = decorated_learner.put_query_process
        self.put_get_labeled = decorated_learner.put_get_labeled
        self.get_init_size = decorated_learner.get_init_size
        self.get_train_size = decorated_learner.get_train_size
        self.get_queries_number = decorated_learner.get_queries_number
        self.get_batch_size = decorated_learner.get_batch_size
        self.get_random_state = decorated_learner.get_random_state


class SbcLearner(AbstractLearnerDecorator):

    def __init__(self, decorated_learner, encoder):
        AbstractLearnerDecorator.__init__(self, decorated_learner)
        self.encoder = encoder
        decorated_learner.put_get_labeled(self.get_labeled_set)

    def get_labeled_set(self, init_size, x, y):
        print("=== preparing clustered labeled data ===")
        return init_generators.cluster_init(x, y, init_size, self.encoder)


class SudLearner(AbstractLearnerDecorator):

    def __init__(self, decorated_learner, encoder):
        AbstractLearnerDecorator.__init__(self, decorated_learner)
        self.encoder = encoder
        decorated_learner.put_query_process(self.query_process)

    def query_process(self, x, y, labeled, x_labeled, y_labeled, val_data, model):
        acc_growth = []

        x_reshaped = x.reshape((len(x), np.prod(x.shape[1:])))
        x_encoded = self.encoder.predict(x_reshaped)
        x_encoded = [e.flatten() for e in x_encoded]

        center_vector = np.sum([x_encoded[i].flatten() for i in range(len(x)) if not labeled[i]], axis=0)
        unlabeled_size = len(x) - len(x_labeled)
        mass_center = center_vector / unlabeled_size

        if val_data is not None:
            acc_growth.append((x_labeled.shape[0], model.evaluate(val_data[0], val_data[1], verbose=0)[1]))
        print("=== SUD learning started === ")
        with progressbar.ProgressBar(maxval=self.get_queries_number(), redirect_stdout=True) as bar:
            for i in range(self.get_queries_number()):
                query = query_entropy_sud(model, x, x_encoded, labeled, self.get_batch_size(), mass_center)
                for j in query:
                    x_labeled = np.append(x_labeled, [x[j]], axis=0)
                    y_labeled = np.append(y_labeled, [y[j]], axis=0)
                    labeled[j] = True
                    center_vector -= x_encoded[j]
                unlabeled_size -= self.get_batch_size()
                mass_center = center_vector / unlabeled_size
                model.fit(x_labeled, y_labeled, validation_data=val_data, verbose=0)
                if val_data is not None:
                    acc_growth.append((x_labeled.shape[0], model.evaluate(val_data[0], val_data[1], verbose=0)[1]))
                bar.update(i)
        self.put_stat(acc_growth)
        print("=== SUD learning finished ===")
