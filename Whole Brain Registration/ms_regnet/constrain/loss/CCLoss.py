from torch import nn
import torch
from ..loss_zoo import LossZoo


@LossZoo.register(
    ("simi", "cc"),
    ("tra", "cc"),
    ("outline", "cc"),
    ("hpf", "cc"),
    ("convex", "cc"),
    ("hole", "cc"),
    ("cp", "cc"),
    ("csc", "cc"),
    ("bs", "cc"),
    ("cbx", "cc"),
    ("ctx", "cc"),
)
class CCLoss(nn.Module):
    def __init__(self, win=[4, 4, 4]):
        super(CCLoss, self).__init__()
        self.win = win

    def forward(self, fix: dict, mov: dict, reg: dict, deform_space: torch.Tensor):
        I, J = fix["img"], reg["img"]
        conv = nn.Conv3d(1, 1, self.win[0], padding=1, bias=False)
        conv.register_parameter(name='weight',
                                param=nn.Parameter(torch.ones([1, 1, self.win[0], self.win[1], self.win[2]])))
        for param in conv.parameters():
            param.requires_grad = False
        conv = conv.to(I.device)
        I2 = I*I
        J2 = J*J
        IJ = I*J

        I_sum = conv(I)
        J_sum = conv(J)
        I2_sum = conv(I2)
        J2_sum = conv(J2)
        IJ_sum = conv(IJ)

        win_size = self.win[0]*self.win[1]*self.win[2]
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var+1e-5)
        cc_loss = -1.0*torch.mean(cc)
        return cc_loss
