"""
store all function used in registration, including fix, mov, deform space
gain a reg dictionary by process the mov data
"""
from typing import Optional

import numpy as np
import torch
from torch.nn.functional import grid_sample

from ..tools.deform import grid_sample_nearest


class DeformerZoo:
    constrain2deformer = {}

    @staticmethod
    def get_deformer_by_constrain(constrain: str):
        return DeformerZoo.constrain2deformer.get(constrain, deform_img_nearest)

    @staticmethod
    def register(*args):
        def inner_register(func):
            for arg in args:
                print(f"add deformer {arg} {func} to deformer zoo")
                DeformerZoo.constrain2deformer[arg] = func
            return func

        return inner_register


@DeformerZoo.register(
                      "simi", "tra", "647"
                      )
def deform_img(fix: dict, mov: dict, deform_space: torch.Tensor) -> dict:
    """
    use the deform space to deform the moving img into registration img
    """
    reg = {"img": grid_sample(mov["img"], deform_space, align_corners=True),
           "img_raw": grid_sample(mov["img_raw"], deform_space, align_corners=True)}
    for k in mov.keys():
        if k not in reg:
            reg[k] = mov[k].clone()
    return reg


@DeformerZoo.register(
                      "outline",
                      "convex",
                      "hpf",
                      "hole",
                      "cp",
                      "csc",
                      "bs",
                      "cbx",
                      "ctx",
                      "cb",
                      "nn"
                      )
def deform_img_nearest(fix: dict, mov: dict, deform_space: torch.Tensor) -> dict:
    """
    use the deform space to deform the moving img into registration img
    """

    reg = {"img_raw": grid_sample(mov["img_raw"], deform_space, mode='nearest', align_corners=True),
           "img": grid_sample(mov["img"], deform_space, align_corners=True)}
    for k in mov.keys():
        if k not in reg:
            reg[k] = mov[k].clone()
    return reg


@DeformerZoo.register("annotation")
def deform_annotation(fix: dict, mov: dict, deform_space: torch.Tensor) -> dict:
    """
    The function named deform_img_nearest cannot be used because there are thousands of labels in
    allen's annotation, sow we cannot treat the annotation as uint16 data, instead of the float32 data.
    So the grid sample cannot be used, so we must implement grid sample by ourselve.
    """

    reg = {"img": grid_sample(mov["img"], deform_space, align_corners=True),
           "img_raw": grid_sample_nearest(mov["img_raw"], deform_space) }
    for k in mov.keys():
        if k not in reg:
            reg[k] = mov[k].clone()
    return reg
