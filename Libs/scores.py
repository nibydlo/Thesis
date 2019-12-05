def score_sequential(model, x_test, y_test):
    return model.evaluate(x_test, y_test, verbose=0)[1]
