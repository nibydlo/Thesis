import learners
import plots
import matplotlib.pyplot as plt
import encoders


simple_learner = learners.EntropyLearner()
sud_learner = learners.SudLearner(simple_learner, encoders.mnist_encoder)
sbc_sud_learner = learners.SbcLearner(sud_learner, encoders.mnist_encoder)
sbc_sud_learner.compile(init_size=2000,
            train_size=10000,
            queries_number=5,
            batch_size=1,
            random_state=0)

import datasets as ds
x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist = ds.get_mnist()

from models import create_sequential_model
model = create_sequential_model()

sbc_sud_learner.learn(model, x_train_mnist, y_train_mnist, validation_data=(x_test_mnist, y_test_mnist))

plots.plot_single(sbc_sud_learner.get_stat(), 'test')
plt.xlabel('labeled set size')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.show()
