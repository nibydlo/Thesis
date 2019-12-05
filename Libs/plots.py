import math
import matplotlib.pyplot as plt


CC = 0.95


def plot_conf_int(stat, stat_name):
    if len(stat) == 0:
        return
    q = len(stat[0])
    n = len(stat)
    mean = [0 for i in range(q)]
    sum_1 = [0 for i in range(q)]
    sum_2 = [0 for i in range(q)]

    for i in range(len(stat)):
        accurs = [e[1] for e in stat[i]]
        for j in range(q):
            mean[j] += accurs[j] / n
            sum_1[j] += accurs[j] ** 2 / n
            sum_2[j] += accurs[j] / n

    D = [sum_1[i] - sum_2[i] ** 2 for i in range(q)]
    sigma = [CC * math.sqrt(d) / math.sqrt(n) for d in D]

    plt.fill_between(
        range(stat[0][0][0], stat[0][-1][0] + 1, stat[0][1][0] - stat[0][0][0]),
        [m + s for (m, s) in zip(mean, sigma)],
        [m - s for (m, s) in zip(mean, sigma)],
        label=stat_name,
        alpha=0.7
    )


def plot_single(accs, name):
    xs_entropy_sbc = [e[0] for e in accs]
    ys_entropy_sbc = [e[1] for e in accs]
    plt.plot(xs_entropy_sbc, ys_entropy_sbc, label=name)
