import math
import numpy as np


def query_default(model, x_train, labeled, batch_size, reshape_f):
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


def query_uncert(model, x_train, labeled, batch_size, uncert_f, reshape_f):
    pre_batch = []
    for i in range(len(x_train)):
        if not labeled[i]:
            p = model.predict_proba(reshape_f(x_train[i]))
            pre_batch.append((uncert_f(p.flatten()), i))
    return [i for (p, i) in sorted(pre_batch)[::-1][:batch_size]]


def query_uncert_1(model, x_train, labeled, batch_size, reshape_f):
    return query_uncert(model, x_train, labeled, batch_size, f1, reshape_f)


def query_uncert_2(model, x_train, labeled, batch_size, reshape_f):
    return query_uncert(model, x_train, labeled, batch_size, f2, reshape_f)


def query_entropy(model, x_train, labeled, batch_size, reshape_f=(lambda x: np.expand_dims(x, axis=0))):
    return query_uncert(model, x_train, labeled, batch_size, f_entropy, reshape_f)


def query_uncert_sud(model, x_train, x_encoded, labeled, batch_size, uncert_f, reshape_f, mass_center):
    pre_batch = []
    for i in range(len(x_train)):
        if not labeled[i]:
            p = model.predict_proba(reshape_f(x_train[i]))
            sim = 1 / np.linalg.norm(x_encoded[i] - mass_center)
            pre_batch.append((uncert_f(p.flatten()) * sim, i))
    return [i for (p, i) in sorted(pre_batch)[::-1][:batch_size]]


def query_uncert_1_sud(model, x_train, x_encoded, labeled, batch_size, reshape_f, mass_center):
    return query_uncert_sud(model, x_train, x_encoded, labeled, batch_size, f1, reshape_f, mass_center)


def query_uncert_2_sud(model, x_train, x_encoded, labeled, batch_size, reshape_f, mass_center):
    return query_uncert_sud(model, x_train, x_encoded, labeled, batch_size, f2, reshape_f, mass_center)


def query_entropy_sud(model, x_train, x_encoded, labeled, batch_size,  mass_center, reshape_f=(lambda x: np.expand_dims(x, axis=0))):
    return query_uncert_sud(model, x_train, x_encoded, labeled, batch_size, f_entropy, reshape_f, mass_center)