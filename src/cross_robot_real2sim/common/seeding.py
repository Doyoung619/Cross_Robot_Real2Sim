import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def seed_split(seed, count):
    return list(range(int(seed) + 10000, int(seed) + 10000 + int(count)))
