import torch
import random
import numpy as np


class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


device = "cuda:0" if torch.cuda.is_available() else "cpu"

if device=='cpu':
    config = DotDict({
        'batch_size': 8,
        'gamma': 2,
        'alpha': 1,
        'n': 1,
        'lr': 1e-3,
        'adam_eps': 1e-08,
        'anchor_num': 64,
    })
else:
    config = DotDict({
        'batch_size': 8,
        'gamma': 2,
        'alpha': 1,
        'n': 1,
        'lr': 1e-3,
        'adam_eps': 1e-08,
        'anchor_num': 64,
    })

nn_config = DotDict({
    'layer_num': 1,
    'embed_dim': 32,
    'feature_dim': 32,
    'hidden_dim': 32,
    'output_dim': 32,
})


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)