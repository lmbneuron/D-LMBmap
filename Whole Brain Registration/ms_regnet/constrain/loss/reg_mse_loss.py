import torch
from torch import nn
from torch.nn.functional import mse_loss
from ..loss_zoo import LossZoo


@LossZoo.register(
                    ("outline", "mse"),
                    ("hpf", "mse"),
                    ("convex", "mse"),
                    ("hole", "mse"),
                    ("simi", "mse"),
                    ("cp", "mse"),
                    ("csc", "mse"),
                    ("bs", "mse"),
                    ("cbx", "mse"),
                    ("ctx", "mse"),
                    ("tra", "mse"),
                    ("cb", "mse"),
                    ("aq", "mse"),
                    ("fx", "mse"),
                    ("fr", "mse"),
                    ("ipn", "mse"),
                    ("mh_lh", "mse"),
                    ("act", "mse"),
                    ("mh_lh_auto", "mse"),
                    ("fr_auto", "mse"),
                    ("fx_auto", "mse"),
                    ("act_auto", "mse"),
                    ("ipn_auto", "mse"),
                   )
class RegMSE(nn.Module):
    def __init__(self):
        super(RegMSE, self).__init__()
        pass

    def forward(self, fix: dict, mov: dict, reg: dict, deform_space: torch.Tensor):
        return mse_loss(reg["img"], fix["img"])
