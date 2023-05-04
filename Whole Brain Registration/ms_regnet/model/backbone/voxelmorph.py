import torch
from torch import nn
from torch.nn.functional import interpolate
from torch import norm, prod, tensor
from torch.nn.functional import mse_loss, grid_sample
import numpy as np

from .BaseModule import BaseModule, conv3d_with_leakyReLU, median_blur
from ...tools.deform import delta2to, generate_base_deform_space


class VoxelMorph(BaseModule):
    def __init__(self,
                 loss: str,
                 scale,
                 cfg: dict,
                 flow_multiplier=1,
                 no_loss: bool = False,
                 no_space: bool = True):
        super(VoxelMorph, self).__init__(loss, cfg, no_loss, no_space)
        self.scale = scale
        self.flow_multiplier = flow_multiplier
        self.median_filter_ksize = cfg["median_filter_ksize"]
        self.max_delta = cfg.get("max_delta", 1)
        self.dsample_module_list_list = nn.ModuleList([
            nn.ModuleList(
                [nn.Sequential(conv3d_with_leakyReLU(self.input_channel, 16, 3, 2, 1),
                               conv3d_with_leakyReLU(16, self.input_channel, 3, 1, 1))
                for _ in range(i)]
            ) for i in range(self.scale)
        ])
        self.encoder_0 = conv3d_with_leakyReLU(self.input_channel, 16, 3, 1, 1)
        self.encoder_1 = conv3d_with_leakyReLU(16, 32, 3, 2, 1)
        self.encoder_2 = conv3d_with_leakyReLU(32, 32, 3, 2, 1)
        self.encoder_3 = conv3d_with_leakyReLU(32, 32, 3, 2, 1)
        self.decoder_0 = conv3d_with_leakyReLU(32, 32, 3, 1, 1)
        self.decoder_1 = conv3d_with_leakyReLU(64, 32, 3, 1, 1)
        self.decoder_2 = conv3d_with_leakyReLU(64, 32, 3, 1, 1)
        self.decoder_3 = conv3d_with_leakyReLU(48, 32, 3, 1, 1)
        self.decoder_4 = conv3d_with_leakyReLU(32, 32, 3, 1, 1)
        self.decoder_5 = conv3d_with_leakyReLU(32+self.input_channel, 16, 3, 1, 1)
        self.decoder_6 = conv3d_with_leakyReLU(16, 16, 3, 1, 1)
        self.decoder_7 = conv3d_with_leakyReLU(16, 3, 3, 1, 1)

        self.usample_module_list_list = nn.ModuleList([
            nn.ModuleList(
                [conv3d_with_leakyReLU(3 + self.input_channel, 3, 3, 1, 1)
                 for _ in range(i)]
            ) for i in range(self.scale)
        ])
        self.act = nn.Tanh()
        self._fix()

    def _fix(self):
        if self.scale > 1:
            for name, param in self.named_parameters():
                if "encoder" in name or "decoder" in name:
                    param.requires_grad = False
            for i in range(self.scale-1):
                for name, param in self.dsample_module_list_list[i].named_parameters():
                    param.requires_grad = False
                for name, param in self.usample_module_list_list[i].named_parameters():
                    param.requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad is False:
                print(f"voxelmorph {name} is fixed")
            else:
                print(f"voxelmorph {name} is not fixed")

    def forward(self, input_dict):
        x = self.fusion_input(input_dict)
        scale_feat_list = []
        for scale in range(self.scale):
            '''down sample'''
            d_sample_feat_list = [x[..., ::1<<(self.scale-scale-1), ::1<<(self.scale-scale-1), ::1<<(self.scale-scale-1)]]
            d_sample_module_list = self.dsample_module_list_list[scale]
            for i, d_layer in enumerate(d_sample_module_list):
                feat = d_layer(d_sample_feat_list[-1])
                d_sample_feat_list.append(feat)
                torch.cuda.empty_cache()
            scale_feat_list.append(d_sample_feat_list)

        x = scale_feat_list[0][-1]
        for i in range(1, len(scale_feat_list)):
            x = x + scale_feat_list[i][-1]

        start = x
        conv_0 = self.encoder_0(x)
        conv_0_shape = np.array(conv_0.shape[2:])
        conv_1 = self.encoder_1(conv_0)
        conv_1_shape = np.array(conv_1.shape[2:])
        conv_2 = self.encoder_2(conv_1)
        conv_2_shape = np.array(conv_2.shape[2:])
        conv_3 = self.encoder_3(conv_2)

        x = self.decoder_0(conv_3)
        x_shape = np.array(x.shape[2:])
        x = torch.cat((interpolate(x, scale_factor=(conv_2_shape/x_shape).tolist(), mode='trilinear'), conv_2), dim=1)
        x = self.decoder_1(x)
        x_shape = np.array(x.shape[2:])
        x = torch.cat((interpolate(x, scale_factor=(conv_1_shape/x_shape).tolist(), mode='trilinear'), conv_1), dim=1)
        x = self.decoder_2(x)
        x_shape = np.array(x.shape[2:])
        x = torch.cat((interpolate(x, scale_factor=(conv_0_shape/x_shape).tolist(), mode='trilinear'), conv_0), dim=1)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x_shape = np.array(x.shape[2:])
        x = torch.cat((x, start), dim=1)
        x = self.decoder_5(x)
        x = self.decoder_6(x)
        x = self.decoder_7(x)
        # x = x * self.flow_multiplier

        output_list = []
        '''up sample'''
        for scale in range(self.scale):
            '''down sample'''
            feat = x
            u_sample_module_list = self.usample_module_list_list[scale]
            for i, d_module in enumerate(u_sample_module_list):
                feat = torch.cat((feat, scale_feat_list[scale][-i-1]), 1)
                feat = d_module[0](feat)
                feat = interpolate(feat, scale_factor=2, mode='trilinear')
                feat = d_module[1](feat)
                torch.cuda.empty_cache()

            if scale < self.scale - 1:
                feat = interpolate(feat, scale_factor=1 << (self.scale-scale-1), mode='trilinear')
            output_list.append(feat)
            torch.cuda.empty_cache()

        x = output_list[0]
        for i in range(1, len(output_list)):
            x = x + output_list[i]
        x = self.act(x)
        torch.cuda.empty_cache()
        reg_loss = self._regularize_loss(x)
        x = median_blur(x, self.median_filter_ksize)
        x *= self.max_delta
        torch.cuda.empty_cache()
        x = x.permute(0, 2, 3, 4, 1)

        delta_deform_space = x



        to_deform_space = delta2to(delta_deform_space, input_dict["fix"]["simi"]["img"].shape)
        to_deform_space = torch.clip(to_deform_space, -1, 1)
        # indent_loss = self._indentity_loss(to_deform_space, input_dict["fix"]["simi"]["img].shape)
        torch.cuda.empty_cache()
        result_dict = {
                        "reg": {},
                        "loss": {
                            "reg_loss": reg_loss,
                            # "indent_loss": indent_loss
                        }}
        self.handle_output(input_dict, to_deform_space, result_dict)


        # 计算反向映射的loss
        delta_deform_space_inv = self.flowinverse(delta_deform_space)
        to_deform_space_inv = delta2to(delta_deform_space_inv, input_dict["fix"]["simi"]["img"].shape)
        reg_inv_dict = self.cal_reg({}, result_dict["reg"], to_deform_space_inv)
        inv_loss = 0
        for k in reg_inv_dict.keys():
            inv_loss += mse_loss(reg_inv_dict[k]["img"], input_dict["mov"][k]["img"])
        result_dict["loss"]["inv_loss"] = inv_loss
        result_dict["inv"] = reg_inv_dict
        torch.cuda.empty_cache()

        return result_dict

    def _regularize_loss(self, flow):
        ret = (norm(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :], p=2)
               + norm(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :], p=2)
               + norm(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1], p=2)) \
              / prod(tensor(flow.shape, device=flow.device))
        return ret

    def _indentity_loss(self, flow, shape):
        base_def_space = generate_base_deform_space(shape, flow.device)
        return mse_loss(flow, base_def_space)

    def flowinverse(self,  flow):
        """
        calculate a reverse deform space, flow should be offset of points, instead of coordinate
        """

        flow0 = torch.unsqueeze(flow[:, :, :, :, 0], 1)
        flow1 = torch.unsqueeze(flow[:, :, :, :, 1], 1)
        flow2 = torch.unsqueeze(flow[:, :, :, :, 2], 1)

        flow0 = grid_sample(flow0, flow, align_corners=True)
        flow1 = grid_sample(flow1, flow, align_corners=True)
        flow2 = grid_sample(flow2, flow, align_corners=True)

        flow_inverse = torch.cat((flow0, flow1, flow2), 1)
        flow_inverse = flow_inverse.permute(0, 2, 3, 4, 1)
        flow_inverse *= -1
        return flow_inverse

if __name__ == "__main__":
    import numpy as np
    net = VoxelMorph(loss='mi', scale=3, use_recon_loss=True, use_outline_loss=True, use_hole_loss=True, flow_multiplier=1)
    input = {"fix":  torch.Tensor(np.random.random_sample((2, 1, 320, 456, 528))).cuda(),
             "fix_outline": torch.Tensor(np.random.random_sample((2, 1, 320, 456, 528))).cuda(),
             "fix_hole": torch.Tensor(np.random.random_sample((2, 1, 320, 456, 528))).cuda(),
             "mov": torch.Tensor(np.random.random_sample((2, 1, 320, 456, 528))).cuda(),
             "mov_outline": torch.Tensor(np.random.random_sample((2, 1, 320, 456, 528))).cuda(),
             "mov_hole": torch.Tensor(np.random.random_sample((2, 1, 320, 456, 528))).cuda()}
    net = net.to('cuda')
    output = net(input)
