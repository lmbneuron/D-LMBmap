"""
define optional function can be used to evaluate registration
both fix and reg are dict
"""

import numpy as np

__all__ = ["MetricZoo"]


class MetricZoo:
    constrain2metric = {}

    @staticmethod
    def get_metric_by_constrain_and_type(constrain: str, type: str=None):
        type = "dice" if type is None else type
        return MetricZoo.constrain2metric[type].get(constrain, None)

    @staticmethod
    def register(name, constraints):
        def inner_register(func):
            MetricZoo.constrain2metric[name] = {}
            for cons in constraints:
                print(f"constrain: {cons} uses the {name} metric {func}")
                MetricZoo.constrain2metric[name][cons] = func
            return func

        return inner_register


@MetricZoo.register("iou",
                    (
                        "outline", "convex", "hpf", "hole",
                        "hole_pointcloud", "cp", "hole_landmark", "csc",
                        "bs", "cbx", "ctx", 'cb', "nn", "fr", "fx", "ipn", "act", "mh_lh", "aq"
                    )
                    )
def iou_metric(fix: dict, reg: dict):
    """
    :param fix: series of data about fix image
    :param reg: image data after registration
    :return:
    """
    fix_img, reg_img = fix["img_raw"].clone().detach().cpu().numpy(), reg["img_raw"].clone().detach().cpu().numpy()
    return cal_iou(fix_img, reg_img)



@MetricZoo.register("dice",
                    (
                        "outline", "convex", "hpf", "hole",
                        "hole_pointcloud", "cp", "hole_landmark", "csc",
                        "bs", "cbx", "ctx", 'cb', "nn", "fr", "fx", "ipn", "act", "mh_lh", "aq"
                    )
                    )
def dice_metric(fix: dict, reg: dict):
    fix_img, reg_img = fix["img_raw"].clone().detach().cpu().numpy(), reg["img_raw"].clone().detach().cpu().numpy()
    return cal_dice(fix_img, reg_img)


@MetricZoo.register("dice4sr",
                    (
                    "fr", "fx", "ipn", "act", "mh_lh"
                    )
                    )
def dice_metric(fix: dict, reg: dict):
    fix_img, reg_img = fix["img"].clone().detach().cpu().numpy(), reg["img"].clone().detach().cpu().numpy()
    return cal_dice(fix_img, reg_img)


def cal_iou(vol1, vol2):
    """
    iou calculation
    0 as background，else are brain region
    """
    vol1[vol1 > 0] = 1
    vol2[vol2 > 0] = 1
    vol1 = vol1.astype(np.uint8)
    vol2 = vol2.astype(np.uint8)
    vol_i = vol2.copy()
    vol_i[vol1 == 0] = 0
    vol_i[vol_i != 0] = 1
    vol_u = vol2.copy()
    vol_u[vol1 != 0] = 1
    vol_u[vol_u != 0] = 1
    i = np.sum(vol_i)
    u = np.sum(vol_u)
    return i / u


def cal_dice(vol1, vol2, thres=0.2):
    """
    dice calculation
    0 as background，else are brain region
    """
    vol1[vol1 <= thres] = 0
    vol2[vol2 <= thres] = 0
    vol1[vol1 > thres] = 1
    vol2[vol2 > thres] = 1
    vol1 = vol1.astype(np.uint8)
    vol2 = vol2.astype(np.uint8)
    vol_i = vol2.copy()
    vol_i[vol1 == 0] = 0
    vol_i[vol_i != 0] = 1
    i = np.sum(vol_i)
    u = np.sum(vol1) + np.sum(vol2)
    return 2*i / u
