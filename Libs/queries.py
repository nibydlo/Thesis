import math
import numpy as np
from enum import Enum


def query_default(_, x_train, labeled, batch_size):
    res = []
    for i in range(len(x_train)):
        if not labeled[i]:
            res += [i]
        if len(res) == batch_size:
            return res
    return res


def f1(p):
    return 1 - max(p)


def f2(p):
    return 1 - (sorted(p)[-1] - sorted(p)[-2])


def f_entropy(p):
    return -1 * sum([e * math.log((e + math.pow(10, -10))) for e in p])


def query_uncert(model, x_train, labeled, batch_size, uncert_f):
    pre_batch = []
    for i in range(len(x_train)):
        if not labeled[i]:
            p = model.predict_proba(np.expand_dims(x_train[i], axis=0))
            pre_batch.append((uncert_f(p.flatten()), i))
    return [i for (p, i) in sorted(pre_batch)[::-1][:batch_size]]


def query_uncert_1(model, x_train, labeled, batch_size):
    return query_uncert(model, x_train, labeled, batch_size, f1)


def query_uncert_2(model, x_train, labeled, batch_size):
    return query_uncert(model, x_train, labeled, batch_size, f2)


def query_entropy(model, x_train, labeled, batch_size):
    return query_uncert(model, x_train, labeled, batch_size, f_entropy)


def query_uncert_sud(model, x_train, x_encoded, labeled, batch_size, uncert_f, mass_center):
    pre_batch = []
    for i in range(len(x_train)):
        if not labeled[i]:
            p = model.predict_proba(np.expand_dims(x_train[i], axis=0))
            sim = 1 / np.linalg.norm(x_encoded[i] - mass_center)
            pre_batch.append((uncert_f(p.flatten()) * sim, i))
    return [i for (p, i) in sorted(pre_batch)[::-1][:batch_size]]


def query_density_entropy(model, x_train, x_encoded, labeled, batch_size, uncert_f, k, b):
    pre_batch = []
    for i in range(len(x_train)):
        if not labeled[i]:
            distances = [(np.linalg.norm(x_encoded[i] - x_encoded[j]), j) for j in range(len(x_train)) if not labeled[j]]
            distances = sorted(distances)
            ds = (1 / (distances[k][0] ** np.shape(x_encoded)[1])) ** b
            p = model.predict_proba(np.expand_dims(x_train[i], axis=0))
            pre_batch.append((uncert_f(p.flatten()) * ds, i))
    return [i for (p, i) in sorted(pre_batch)[::-1][:batch_size]]


def query_uncert_1_sud(model, x_train, x_encoded, labeled, batch_size, mass_center):
    return query_uncert_sud(model, x_train, x_encoded, labeled, batch_size, f1, mass_center)


def query_uncert_2_sud(model, x_train, x_encoded, labeled, batch_size, mass_center):
    return query_uncert_sud(model, x_train, x_encoded, labeled, batch_size, f2, mass_center)


def query_entropy_sud(model, x_train, x_encoded, labeled, batch_size, mass_center):
    return query_uncert_sud(model, x_train, x_encoded, labeled, batch_size, f_entropy, mass_center)


# def query_entropy_sud(model, x_train, x_encoded, labeled, batch_size, k, b):
#     return query_density_entropy(model, x_train, x_encoded, labeled, batch_size, f_entropy, k, b)


class Queries(Enum):
    default = query_default
    uncert_1 = query_uncert_1
    uncert_2 = query_uncert_2
    entropy = query_entropy
    uncert_1_sud = query_uncert_1
    uncert_2_sud = query_uncert_2
    entropy_sud = query_entropy_sud

    def get_by_name(name):
        if name == 'default':
            return Queries.default
        if name == 'uncert_1':
            return Queries.uncert_1
        if name == 'uncert_2':
            return Queries.uncert_2
        if name == 'entropy':
            return Queries.entropy
        if name == 'uncert_1_sud':
            return Queries.uncert_1_sud
        if name == 'uncert_2_sud':
            return Queries.uncert_2_sud
        if name == 'entropy_sud':
            return Queries.entropy_sud
        return None
