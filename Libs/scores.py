from enum import Enum


def score_sequential(model, val_data):
    x_test, y_test = val_data
    return model.evaluate(x_test, y_test, verbose=0)[1]


class Scores(Enum):
    accuracy = score_sequential

    def get_by_name(name):
        if name == 'accuracy':
            return Scores.accuracy
        return None
