from torch import nn
import torch
from torch.nn.functional import affine_grid, adaptive_avg_pool3d, tanh, mse_loss
from torch import sin, cos

from .BaseModule import conv3d_with_leakyReLU, BaseModule


class Rigid(BaseModule):
    def __init__(self,
                 loss: str,
                 scale,
                 cfg: dict,
                 no_loss: bool = False,
                 no_space: bool = True):
        super(Rigid, self).__init__(loss, cfg, no_loss, no_space)
        self.scale = scale
        self.conv_list = nn.Sequential(
            conv3d_with_leakyReLU(self.input_channel, 16, 3, 1, 1),
            conv3d_with_leakyReLU(16, 32, 3, 2, 1),
            conv3d_with_leakyReLU(32, 64, 3, 1, 1),
            conv3d_with_leakyReLU(64, 128, 3, 2, 1),
            conv3d_with_leakyReLU(128, 128, 3, 1, 1),
            conv3d_with_leakyReLU(128, 256, 3, 2, 1),
            conv3d_with_leakyReLU(256, 256, 3, 1, 1),
            conv3d_with_leakyReLU(256, 512, 3, 2, 1),
            conv3d_with_leakyReLU(512, 512, 3, 1, 1),
        )

        self.conv_w = nn.Conv3d(512, 3, 3, 1)
        self.conv_b = nn.Conv3d(512, 3, 3, 1)
        self._fix()

    def _fix(self):
        if self.scale > 1:
            for name, param in self.named_parameters():
                param.requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad is False:
                print(f"rigid {name} is fixed")
            else:
                print(f"rigid {name} is not fixed")

    def forward(self, input_dict):
        """
        :param input_dict: {
                           fix:  {
                                    simi: {}
                                    outline: {}
                                    hole_pointcloud: {}
                                    convex: {}
                                  }.
                           mov:  {
                                    simi: {}
                                    outline: {}
                                    hole_pointcloud: {}
                                    convex: {}
                                  }
                           }
        :return: {
                    reg:  {
                            simi: {}
                            outline: {}
                            hole_pointcloud: {}
                            convex: {}
                          }
                   },
                   loss: {}
                }
        """
        x = self.fusion_input(input_dict)
        x = x[:, :, ::1<<(self.scale-1), ::1<<(self.scale-1), ::1<<(self.scale-1)]
        x = self.conv_list(x)  # 4*4*4
        torch.cuda.empty_cache()
        w = self.conv_w(x)
        b = self.conv_b(x)
        w = adaptive_avg_pool3d(w, (1, 1, 1))  ##(b, c, x, y, z)
        b = adaptive_avg_pool3d(b, (1, 1, 1))
        w = tanh(w)
        b = tanh(b)
        w, b = w.view([-1, 3]), b.view([-1, 3, 1])

        A_0 = cos(w[:, 0:1])*cos(w[:, 1:2])
        A_1 = cos(w[:, 0:1])*sin(w[:, 1:2])*sin(w[:, 2:])-sin(w[:, 0:1])*cos(w[:, 2:])
        A_2 = cos(w[:, 0:1])*sin(w[:, 1:2])*cos(w[:, 2:])+sin(w[:, 0:1])*sin(w[:, 2:])
        A_012 = torch.cat((A_0, A_1, A_2), 1)
        A_012 = A_012.view((-1, 1, 3))

        A_3 = sin(w[:, 0:1])*cos(w[:, 1:2])
        A_4 = sin(w[:, 0:1])*sin(w[:, 1:2])*sin(w[:, 2:])+cos(w[:, 0:1])*cos(w[:, 2:])
        A_5 = sin(w[:, 0:1])*sin(w[:, 1:2])*cos(w[:, 2:])-cos(w[:, 0:1])*sin(w[:, 2:])
        A_345 = torch.cat((A_3, A_4, A_5), 1)
        A_345 = A_345.view((-1, 1, 3))

        A_6 = -sin(w[:, 1:2])
        A_7 = -cos(w[:, 1:2])*sin(w[:, 2:])
        A_8 =  cos(w[:, 1:2])*cos(w[:, 2:])
        A_678 = torch.cat((A_6, A_7, A_8), 1)
        A_678 = A_678.view((-1, 1, 3))

        A = torch.cat((A_012, A_345, A_678), 1)
        deform_matrix = torch.cat((A, b), -1)
        deform_space = affine_grid(deform_matrix, size=input_dict["fix"]["simi"]["img"].shape)
        deform_space = torch.clip(deform_space, -1, 1)

        torch.cuda.empty_cache()
        indent_loss = self._indent_loss(A, b)
        result_dict = {
                            "reg": {},
                            "loss": {
                                "indent_loss": indent_loss
                            }
                        }
        self.handle_output(input_dict, deform_space, result_dict)
        return result_dict

    def _indent_loss(self, A, b):
        batch = A.shape[0]
        I = torch.tile(torch.eye(3, device=A.device).view(1, 3, 3), (batch, 1, 1))
        zero = torch.zeros(b.shape, device=b.device)
        return mse_loss(A, I) + mse_loss(b, zero)