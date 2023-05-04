import os
import pickle
import torch
from torch import nn
from torch.nn.functional import tanh, mse_loss, adaptive_avg_pool3d, pad
import numpy as np
from math import ceil
from tqdm import tqdm

from .BaseModule import BaseModule, conv3d_with_leakyReLU
from ...tools.deform import delta2to, generate_base_deform_space


class Bezier(BaseModule):
    def __init__(self,
                 loss: str,
                 scale,
                 cfg: dict,
                 no_loss: bool = False,
                 no_space: bool = True):
        super(Bezier, self).__init__(loss, cfg, no_loss, no_space)
        self.scale = scale

        self.spline_scale = cfg["scale"]
        self.grid_space = cfg["grid_space"]
        self.max_delta = cfg.get("max_delta", 1)
        self.pred_mode = cfg.get("pred_mode", 0)

        self.cpid_to_select = None
        self.bd = None
        self.num_cpoints = None

        self.dsample_module_list_list = nn.ModuleList(
            [nn.Sequential(conv3d_with_leakyReLU(self.input_channel, 16, 3, 2, 1),
                conv3d_with_leakyReLU(16, self.input_channel, 3, 1, 1))
                for _ in range(self.scale)]
        )

        self.conv = nn.Sequential(
            conv3d_with_leakyReLU(self.input_channel, 16, 3, 1, 1),
            conv3d_with_leakyReLU(16, 16, 3, 1, 1),
            conv3d_with_leakyReLU(16, 32, 3, 2, 1),
            conv3d_with_leakyReLU(32, 32, 3, 1, 1),
            conv3d_with_leakyReLU(32, 32, 3, 2, 1)
        )
        self.last_conv = nn.Conv3d(32, 3, 3, 1, 1)
        self._fix()

    def _fix(self):
        ...

    def _register_data(self, shape, dtype, device):
        assert len(shape) == 3

        file_path = f"myresource/{shape}.pkl"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                cpid_to_select = data["cpid_to_select"]
                bd = data["bd"]
                self.cpt_id_in_ori_image = data["cpt_id_in_ori_image"]
                self.num_cpoints = data["num_cpoints"]
        else:
            bd = []
            cpid_to_select = []
            delta = (self.spline_scale - 1) * self.grid_space

            c_id_2_ori_id = [{} for _ in range(3)]

            for x in tqdm(range(shape[0])):
                if x % delta == 0:
                    cur_cx = x
                    if x + delta < shape[0]:
                        nex_cx = x + delta
                    else:
                        nex_cx = shape[0] - 1
                    for ii in range(self.spline_scale):
                        c_id_2_ori_id[0][cur_cx // self.grid_space + ii] = cur_cx + \
                                                                        int(round((nex_cx - cur_cx)/(self.spline_scale-1)*ii))

                u = (x - cur_cx) / (nex_cx - cur_cx)
                i = cur_cx // self.grid_space

                for y in range(shape[1]):
                    if y % delta == 0:
                        cur_cy = y
                        if y + delta < shape[1]:
                            nex_cy = y + delta
                        else:
                            nex_cy = shape[1] - 1
                        for ii in range(self.spline_scale):
                            c_id_2_ori_id[1][cur_cy // self.grid_space + ii] = cur_cy + \
                                                                            int(round((nex_cy - cur_cy) / (
                                                                                        self.spline_scale - 1) * ii))
                    v = (y - cur_cy) / (nex_cy - cur_cy)
                    j = cur_cy // self.grid_space
                    for z in range(shape[2]):
                        if z % delta == 0:
                            cur_cz = z
                            if z + delta < shape[2]:
                                nex_cz = z + delta
                            else:
                                nex_cz = shape[2] - 1
                            for ii in range(self.spline_scale):
                                c_id_2_ori_id[2][cur_cz // self.grid_space + ii] = cur_cz + \
                                                                                int(round((nex_cz - cur_cz) / (
                                                                                            self.spline_scale - 1) * ii))
                        w = (z - cur_cz) / (nex_cz - cur_cz)
                        k = cur_cz // self.grid_space
                        bd.append(self.bspline(u, v, w))
                        cpid_to_select += self.select_id(i, j, k)
            cpid_to_select = np.array(cpid_to_select, dtype=np.int)
            self.num_cpoints = np.max(cpid_to_select, 0) + 1
            cpid_to_select = cpid_to_select[:, 0] * self.num_cpoints[1] * self.num_cpoints[2] \
                    + cpid_to_select[:, 1] * self.num_cpoints[2]\
                    + cpid_to_select[:, 2]
            # control point image map to original image
            cpt_id_in_ori_image = [[] for _ in range(3)]
            for i in range(3):
                for ii in range(self.num_cpoints[i]):
                    cpt_id_in_ori_image[i].append(c_id_2_ori_id[i][ii])

            cpid_to_select = np.reshape(cpid_to_select, (-1,))
            bd = np.reshape(bd, newshape=(1, 1, shape[0], shape[1], shape[2], -1))
            self.cpt_id_in_ori_image = cpt_id_in_ori_image

            if not os.path.exists(os.path.split(file_path)[0]):
                os.mkdir(os.path.split(file_path)[0])
            with open(file_path, "wb") as f:
                pickle.dump({"cpid_to_select": cpid_to_select,
                             "bd": bd,
                             "cpt_id_in_ori_image": cpt_id_in_ori_image,
                             "num_cpoints": self.num_cpoints
                             }, f)

        self.cpid_to_select = torch.tensor(cpid_to_select, dtype=torch.int32, device=device, requires_grad=False)
        self.bd = torch.tensor(bd, dtype=dtype, device=device, requires_grad=False)

    def forward(self, input_dict):
        x = self.fusion_input(input_dict)

        # pad to the integer multiple of (self.spline_scale-1) * self.grid_space
        # delta = (self.spline_scale-1) * self.grid_space
        # x = pad(x, (0, delta - x.shape[-1] % delta, 0, delta - x.shape[-2] % delta, 0, delta - x.shape[-3] % delta),
        #         'constant', 0)
        # dst_shape = []
        # for i in range(1, 4):
        #     if x.shape[-i] % delta == 0:
        #         dst_shape.append(0)
        #     else:
        #         dst_shape.append(delta - x.shape[-i]%delta)
        # x = pad(x, (0, dst_shape[0], 0, dst_shape[1], 0, dst_shape[2]), 'constant', 0)

        b = x.shape[0]
        shape = [x.shape[2], x.shape[3], x.shape[4]]
        if self.bd is None:
            self._register_data(shape, x.dtype, x.device)

        for module in self.dsample_module_list_list:
            x = module(x)

        x = self.conv(x)
        x = self.last_conv(x)
        x = adaptive_avg_pool3d(x, self.num_cpoints)
        x = tanh(x)
        x = x * self.max_delta
        if self.pred_mode == 1:
            base_space = generate_base_deform_space([b, 1, shape[0], shape[1], shape[2]], x.device)
            base_space = base_space[:, self.cpt_id_in_ori_image[0], ...]
            base_space = base_space[:, :, self.cpt_id_in_ori_image[1], ...]
            base_space = base_space[:, :, :, self.cpt_id_in_ori_image[2], :]
            base_space = base_space.permute(0, 4, 1, 2, 3)
            x = x + base_space

        x = x.view(b, 3, -1)
        x = torch.index_select(x, 2, self.cpid_to_select)
        x = x.view(b, 3, shape[0], shape[1], shape[2], -1)

        x *= self.bd
        x = torch.sum(x, -1, dtype=x.dtype)
        x = x.permute(0, 2, 3, 4, 1)
        if self.pred_mode == 0:
            x = delta2to(x, (1, 1, shape[0], shape[1], shape[2]))
        x = torch.clip(x, -1, 1)
        torch.cuda.empty_cache()
        result_dict = {
            "reg": {},
            "loss": {}
        }
        self.handle_output(input_dict, x, result_dict)
        return result_dict

    def _indentity_loss(self, flow, shape):
        base_def_space = generate_base_deform_space(shape, flow.device)
        return mse_loss(flow, base_def_space)

    def bspline(self, u, v, w):
        dim = self.spline_scale
        bdx = np.zeros([dim, 1])
        for id in range(dim):
            bdx[id] = self.spline(u, id)  # normalized distance

        bdy = np.zeros([dim, 1])
        for id in range(dim):
            bdy[id] = self.spline(v, id)  # normalized distance


        bdz = np.zeros([dim, 1])
        for id in range(dim):
            bdz[id] = self.spline(w, id)  # normalized distance

        bd = np.dot(bdy, bdz.T)
        bd = np.reshape(bd, (1, -1))
        bd = np.dot(bdx, bd)
        bd = np.reshape(bd, (-1,))
        return bd

    def select_id(self, i, j, k):
        sid = []
        for l in range(self.spline_scale):
            for m in range(self.spline_scale):
                for n in range(self.spline_scale):
                    sid.append([i+l, j+m, k+n])
        return sid

    def spline(self, u, d):
        if self.spline_scale == 3:
            if (d == 0):
                return (1-u)**2
            elif d == 1:
                return 2 * u - 2 * u**2
            elif d == 2:
                return u**2
        elif self.spline_scale == 4:
            if (d == 0):
                return (1-u)**3
            elif d == 1:
                return 3 * (u ** 3) - 6 * (u ** 2) + 3*u
            elif d == 2:
                return 3 * (u ** 2) - 3 * (u ** 3)
            elif d == 3:
                return u ** 3
        elif self.spline_scale == 2:
            if d == 0:
                return 1 - u
            else:
                return u
