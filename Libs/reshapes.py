img_rows = img_cols = 28


def reshape_mnist(x):
    return x.reshape(1, img_rows, img_cols, 1)


def reshape_svc(x):
    return x.reshape(1, -1)
