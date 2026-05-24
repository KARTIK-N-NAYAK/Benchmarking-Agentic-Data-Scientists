import os
import torch.optim
from hebo.design_space.design_space import DesignSpace

TAB_EMBED_LR = 1e-4
TAB_HEAD_LR = 1e-4
IMG_EMBED_LR = 1e-5
IMG_HEAD_LR = 1e-5
TXT_EMBED_LR = 1e-5
TXT_HEAD_LR = 1e-5

EFFECTIVE_TRAIN_BATCH_SIZE = 64
EFFECTIVE_TEST_BATCH_SIZE = 64
ACCUMULATE_GRAD_BATCHES = 1

NUM_WORKERS = 4

OPTIMIZER = torch.optim.Adam

HEBO_SPACE = DesignSpace().parse([
    {'name': 'learning_rate', 'type': 'pow', 'lb': 1e-6, 'ub': 1e-2},
    {'name': 'optimizer', 'type': 'cat', 'categories': ['adam', 'sgd', 'adamw']}
])

N_TRIALS = 20
TTA_ROUNDS = 4
NUM_FOLDS = 5
MAX_EPOCHS = 16
MAX_TIME = "00:10:00:00"
TRIALS_DIR = "./trials"
CV_TRIALS_DIR = "./cv_folds"

if os.getenv("AGENT_DEBUG", False) in ["True", "true", "1"]:
    MAX_EPOCHS = 1
    N_TRIALS = 0
    MAX_TIME = "00:00:01:00"  # 1 minute
    NUM_FOLDS = 2
elif str(os.getenv("AGENT_NO_BO", False)) in ["True", "true", "1"]:
    N_TRIALS = 0
    MAX_EPOCHS = 30