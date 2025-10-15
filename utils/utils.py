import os,sys

import random
import torch
import numpy as np


def print_and_log_to_file(log_path, log_data):
    with open(log_path, 'a') as f:
        if isinstance(log_data, str):
            print(log_data)
            print(log_data, file=f)
        elif isinstance(log_data, dict):
            for key, value in log_data.items():
                print(key+value)
                print(key+value, file=f)
        else:
            raise TypeError(f"Expected log_data to be str or dict, got {type(log_data)}")
       

def seed_torch(seed=42):
    """
    set the random seeds
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True