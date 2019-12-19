import scenarios
from queries import Queries
import datasets as ds
import plots
import matplotlib.pyplot as plt

config_dict = {
    'score': 'accuracy',
    'query': 'entropy',
    'model': 'mnist_1',
    'init_size': 2000,
    'batch_size': 1,
    'total_size': 10000,
    'queries_number': 10,
    'random_state':0
}

config = scenarios.Config(config_dict)
print(config)
x, y, x_test, y_test = ds.get_mnist()
experiment = scenarios.Experiment(config, 'uncertainty', x, y, (x_test, y_test))
unc = experiment.run()
plots.plot_single(unc, 'unc')

config_2 = config.set_query_f('default')
experiment_2 = scenarios.Experiment(config_2, 'uncertainty', x, y, (x_test, y_test))
passive = experiment_2.run()
plots.plot_single(passive, 'passive')

config_3 = config_2.set_query_f('entropy').set_encoder('mnist_2')
experiment_3 = scenarios.Experiment(config_3, 'sbc', x, y, (x_test, y_test))
sbc = experiment_3.run()
plots.plot_single(sbc, 'sbc')

plt.legend()
plt.show()