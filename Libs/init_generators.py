from sklearn.cluster import AgglomerativeClustering
import numpy as np


def cluster_init(x, y, init_size, encoder):
    x_reshaped = x.reshape((len(x), np.prod(x.shape[1:])))
    x_encoded = encoder.predict(x_reshaped)
    x_encoded = [e.flatten() for e in x_encoded]

    clustering = AgglomerativeClustering(n_clusters=init_size).fit(x_encoded)
    vec_sums = [np.zeros(len(x_encoded[0])) for j in range(init_size)]
    vec_nums = [0.0 for j in range(init_size)]
    for i in range(len(x)):
        vec_sums[clustering.labels_[i]] = np.add(vec_sums[clustering.labels_[i]], x_encoded[i])
        vec_nums[clustering.labels_[i]] += 1

    centers = [a / b for (a, b) in zip(vec_sums, vec_nums)]
    closest = [None for _ in range(init_size)]
    for i in range(len(x)):
        label = clustering.labels_[i]
        d = np.linalg.norm(x_encoded[i] - centers[label])
        if closest[label] is None or d < closest[label][1]:
            closest[label] = (i, d)

    is_init = [False for _ in range(len(x))]
    for c in closest:
        is_init[c[0]] = True
    x_labeled, y_labeled = extract_labeled(is_init, x, y)
    return is_init, x_labeled, y_labeled


def default_init(x, y, init_size):
    is_init = [True if i < init_size else False for i in range(len(x))]
    x_labeled, y_labeled = extract_labeled(is_init, x, y)
    return is_init, x_labeled, y_labeled


def extract_labeled(is_labeled, x, y):
    x = np.array([x[i] for i in range(len(x)) if is_labeled[i]])
    y = np.array([y[i] for i in range(len(y)) if is_labeled[i]])
    return x, y
