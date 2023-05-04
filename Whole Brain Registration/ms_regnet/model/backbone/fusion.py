"""
methods to fuse different constrain before every module
"""

import torch
from torch import nn
from math import log


class FusionFactory:
    @staticmethod
    def create_fusion_module(cfg: dict):
        fusion_type = cfg.get("fusion_type", 0)
        if fusion_type == 0:
            print("fusion using concat fusion")
            return ConcatFusion(cfg)
        elif fusion_type == 1:
            print("fusion using combine fusion")
            return CombineFusion(cfg)
        else:
            raise AssertionError
        

class BaseFusion(nn.Module):
    def __init__(self, cfg):
        super(BaseFusion, self).__init__()
        self.cfg = cfg
        self.constrain_dict = self.cfg["constrain"]

    def get_input_channel(self):
        """
        gain channel numbers after fusion
        """
        ...

    def fusion_input(self, input_dict):
        ...


class ConcatFusion(BaseFusion):
    def __init__(self, cfg: dict):
        super(ConcatFusion, self).__init__(cfg)

    def get_input_channel(self):
        in_c = len([v for v in self.constrain_dict.values() if v])
        return in_c * 2

    def fusion_input(self, input_dict):
        input = []
        for k, v in self.constrain_dict.items():
            if v:
                input.append(input_dict["fix"][k]["img"])
                input.append(input_dict["mov"][k]["img"])
        x = torch.cat(input, dim=1)
        return x


class CombineFusion(BaseFusion):
    def __init__(self, cfg: dict):
        super(CombineFusion, self).__init__(cfg)

    def get_input_channel(self):
        return 4

    def fusion_input(self, input_dict):
        input = [input_dict["fix"]["simi"]["img"], input_dict["mov"]["simi"]["img"]]
        keys = [k for k in input_dict["fix"].keys() if k != "simi"]  # get all constrain except simi
        key2label = {k: (i+1)/len(keys) for i, k in enumerate(keys)}  # redefine label of constrain to distinguish
        for img_type in ["fix", "mov"]:
            x = torch.zeros(size=input_dict["fix"]["simi"]["img"].shape,
                            dtype=input_dict["fix"]["simi"]["img"].dtype,
                            device=input_dict["fix"]["simi"]["img"].device)
            for k, v in self.constrain_dict.items():
                if k == "simi" or (not v):
                    continue
                x[input_dict[img_type][k]["img"] > 0] = key2label[k]
            input.append(x)
        x = torch.cat(input, dim=1)
        return x
