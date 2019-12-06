from sklearn.utils import shuffle


def cut_ds(short_size, x, y, random_s):
    x_shuffled, y_shuffled = shuffle(x, y, random_state=random_s)
    if len(x_shuffled) > short_size:
        x_shuffled = x_shuffled[:short_size]
    if len(y_shuffled) > short_size:
        y_shuffled = y_shuffled[:short_size]
    return x_shuffled, y_shuffled
