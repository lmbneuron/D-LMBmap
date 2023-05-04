import torch
from torch import nn
from ..loss_zoo import LossZoo


@LossZoo.register(
    ("outline", "dice"),
    ("hpf", "dice"),
    ("convex", "dice"),
    ("hole", "dice"),
    ("simi", "mse"),
    ("cp", "dice"),
    ("csc", "dice"),
    ("bs", "dice"),
    ("cbx", "dice"),
    ("ctx", "dice"),
    ("act", "dice"),
    ("mh_lh", "dice"),
    ("ipn", "dice"),
    ("fr", "dice"),
    ("fx", "dice"),
    ("mh_lh_auto", "dice"),
    ("fr_auto", "dice"),
    ("fx_auto", "dice"),
    ("act_auto", "dice"),
    ("ipn_auto", "dice"),
)
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, fix: dict, mov: dict, reg: dict, deform_space: torch.Tensor):
        b = fix["img"].shape[0]
        numerator = torch.sum(fix["img"] * reg["img"], dim=(1, 2, 3, 4)) * 2
        denominator = torch.sum(fix["img"], dim=(1, 2, 3, 4)) + torch.sum(reg["img"], dim=(1, 2, 3, 4)) + 1e-6
        return torch.sum(1 - numerator / denominator) / b