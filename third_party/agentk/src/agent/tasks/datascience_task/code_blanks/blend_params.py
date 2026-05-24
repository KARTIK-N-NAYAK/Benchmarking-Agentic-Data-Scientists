from hebo.design_space.design_space import DesignSpace

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

NUM_WORKERS = 4
N_TRIALS = 20
TRIALS_DIR = "./trials"
MAX_EPOCHS = 5

HEBO_SPACE = DesignSpace().parse([
    {'name': 'learning_rate', 'type': 'pow', 'lb': 1e-6, 'ub': 1e-2},
    {'name': 'optimizer', 'type': 'cat', 'categories': ['adam', 'sgd', 'adamw']},
    # {'name': 'dropout', 'type': 'num', 'lb': 0.0, 'ub': 0.6}
])
