"""
Module for global random seed setup to ensure experiment reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds across all libraries (random, numpy, torch, environment).

    Args:
        seed (int): The seed value to be set. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU environments
    
    # Enforce deterministic behavior in PyTorch CUDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Global random seed set to: {seed}")