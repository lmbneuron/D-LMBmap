from torch import nn
import torch
from torch.nn.functional import affine_grid, adaptive_avg_pool3d
from .BaseModule import conv3d_with_leakyReLU, BaseModule
from torch.nn.functional import mse_loss


class VTNAffineModule(BaseModule):
    def __init__(self,
                 loss: str,
                 scale,
                 cfg: dict,
                 no_loss: bool = False,
                 no_space: bool = True):
        super(VTNAffineModule, self).__init__(loss, cfg, no_loss, no_space)
        self.scale = scale
        self.conv_list = nn.Sequential(
            conv3d_with_leakyReLU(self.input_channel, 16, 3, 2, 1),
            conv3d_with_leakyReLU(16, 32, 3, 2, 1),
            conv3d_with_leakyReLU(32, 64, 3, 1, 1),
            conv3d_with_leakyReLU(64, 128, 3, 2, 1),
            conv3d_with_leakyReLU(128, 128, 3, 1, 1),
            conv3d_with_leakyReLU(128, 256, 3, 2, 1),
            conv3d_with_leakyReLU(256, 256, 3, 1, 1),
            conv3d_with_leakyReLU(256, 512, 3, 2, 1),
            conv3d_with_leakyReLU(512, 512, 3, 1, 1),
        )
        self.conv_w = nn.Conv3d(512, 9, 3, 1)
        self.conv_b = nn.Conv3d(512, 3, 3, 1)
        self._fix()

    def _fix(self):
        if self.scale > 1:
            for name, param in self.named_parameters():
                param.requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad is False:
                print(f"affine {name} is fixed")
            else:
                print(f"affine {name} is not fixed")

    def forward(self, input_dict):
        '''
        :param input_dict: {
                           fix:  {
                                    simi: {img: torch.Tensor, ...}
                                    outline: {img: torch.Tensor, ...}
                                    hole_pointcloud: {img: torch.Tensor, ...}
                                    convex: {img: torch.Tensor, ...}
                                  }
                           mov:  {
                                    simi: {img: torch.Tensor, ...}
                                    outline: {img: torch.Tensor, ...}
                                    hole_pointcloud: {img: torch.Tensor, ...}
                                    convex: {img: torch.Tensor, ...}
                                  }
                           }
        :return: {
                    reg:  {
                            simi: {img: torch.Tensor, ...}
                            outline: {img: torch.Tensor, ...}
                            hole_pointcloud: {img: torch.Tensor, ...}
                            convex: {img: torch.Tensor, ...}
                    }
                    loss: {}
                 }
        '''
        x = self.fusion_input(input_dict)
        x = x[:, :, ::1<<(self.scale-1), ::1<<(self.scale-1), ::1<<(self.scale-1)]
        x = self.conv_list(x)  # 4*4*4
        torch.cuda.empty_cache()
        w = self.conv_w(x)
        b = self.conv_b(x)
        w = adaptive_avg_pool3d(w, (1, 1, 1))
        b = adaptive_avg_pool3d(b, (1, 1, 1))
        w, b = w.view([-1, 3, 3]), b.view([-1, 3, 1])
        A = w + torch.eye(3).unsqueeze(0).to(w.device)
        affine_matrix = torch.cat((A, b), -1) ##(-1, 3, 4)

        indent_loss = self._indent_loss(A, b)
        deform_space = affine_grid(affine_matrix, size=input_dict["fix"]["simi"]["img"].shape)
        deform_space = torch.clip(deform_space, -1, 1)
        torch.cuda.empty_cache()
        det_loss = torch.norm(torch.det(A)-1)

        torch.cuda.empty_cache()
        result_dict = {
            "reg": {},
            "loss": {
                "det_loss": det_loss,
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