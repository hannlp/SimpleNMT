# import simplenmt.models
# import simplenmt.train
# import simplenmt.translate
# import simplenmt.utils
# import simplenmt.data
import random
import torch
import numpy as np


__version__ = "0.1"

def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True