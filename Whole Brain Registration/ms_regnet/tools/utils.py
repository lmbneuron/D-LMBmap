import torch
import os
import numpy as np
import random

class ModelSaver:
    def __init__(self, max_save_num):
        """
        :param max_save_num: max checkpoint number to save
        """
        self.save_path_list = []
        self.max_save_num = max_save_num

    def save(self, path, state_dict):
        self.save_path_list.append(path)
        if len(self.save_path_list) > self.max_save_num:
            top = self.save_path_list.pop(0)
            os.remove(top)
        torch.save(state_dict, path)


def set_random_seed(seed=0):
    SEED = seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
