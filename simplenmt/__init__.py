# import simplenmt.models
# import simplenmt.train
# import simplenmt.translate
# import simplenmt.utils
# import simplenmt.data
import random
import torch
import numpy as np


__version__ = "0.1"

def set_seed(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True