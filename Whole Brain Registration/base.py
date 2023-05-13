import os
import torch
import numpy as np
from munch import Munch
from copy import deepcopy
from typing import List
from torch.nn.functional import grid_sample, interpolate

from ms_regnet.tools.io import write_tiff_stack
from ms_regnet.constrain import MetricZoo, restore_raw_image_from_output
from ms_regnet.datasets import Brain, TwoTemplate
from ms_regnet.model.CascadedNetwork import CascadedNet
from ms_regnet.core.tensor_dict import tensor_cpu
from ms_regnet.constrain import DeformerZoo


class Baser:
    def __init__(self):
        ...

    @staticmethod
    def cal_metric(reg: dict, fix: dict, metric_cfg: dict):
        metric_result_dict = {}
        for k in reg.keys():
            if metric_cfg is None:
                metric_calculator = MetricZoo.get_metric_by_constrain_and_type(k)
            else:
                metric_calculator = MetricZoo.get_metric_by_constrain_and_type(k, metric_cfg.get(k))
            if metric_calculator is not None:
                metric_result_dict[k] = metric_calculator(fix[k], reg[k])
        return metric_result_dict

    @staticmethod
    def write_data(data, name, basedir: str, parent_name=None):
        if parent_name is not None:
            basedir = os.path.join(basedir, parent_name)
        basedir = os.path.join(basedir, name)
        if not os.path.exists(basedir):
            os.makedirs(basedir)

        for k in data.keys():
            if k == 'simi':
                write_tiff_stack(restore_raw_image_from_output(data[k]), os.path.join(basedir, name + ".tiff"))
            else:
                write_tiff_stack(restore_raw_image_from_output(data[k]), os.path.join(basedir, f"{name}_{k}.tiff"))

    @staticmethod
    def _get_model(cfg: dict, no_loss=False, no_space=True, checkpoint=None, eval=False):
        checkpoint = cfg["TrainConfig"]["checkpoint"] if checkpoint is None else checkpoint
        model_config = Munch(cfg["ModelConfig"])
        model = CascadedNet(model_config.rigid, model_config.affine, model_config.backbone,
                            loss=cfg["LossConfig"]["loss"],
                            scale=model_config.scale,
                            no_space=no_space, no_loss=no_loss, checkpoint=checkpoint)
        model = model.cuda()
        if eval:
            model.eval()
        return model

    @staticmethod
    def _get_loader(json_path: str,
                    cfg: dict,
                    shuffle=True,
                    batch_size=None):
        print("data_path:", json_path)
        train_type = cfg["TrainConfig"].get("train_type", 0)
        batch_size = cfg["TrainConfig"]["batch"] if batch_size is None else batch_size
        assert train_type in [0, 1], "train type must be 0 or 1"
        if train_type == 0:
            # training mode that only one moving img is used
            dataset = Brain(json_path, constrain=cfg["DataConfig"]["constrain"])
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        elif train_type == 1:
            # training mode that use random moving img and fixed img, used in average brain template training
            dataset = TwoTemplate(json_path, constrain=cfg["DataConfig"]["constrain"])
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    @staticmethod
    def update_reg_2_new_input(reg, input):
        reg = tensor_cpu(reg)
        reg = deepcopy(reg)
        fix = tensor_cpu(input["fix"])
        output = {"fix": fix,
                  "mov": reg}
        return output

    @staticmethod
    def deform(fix_data: dict, mov_data: dict, spaces: List, need_upsample=False, upsample_time=1):
        """
        :param need_upsample: when spaces and fix_data are not the same scale, the spaces can be upsample two times.
        """
        fix_data = deepcopy(fix_data)
        mov_data = deepcopy(mov_data)

        for i in range(len(spaces)):
            space = torch.Tensor(spaces[i])

            if need_upsample:
                space = space.permute(0, 4, 1, 2, 3)
                space = interpolate(space, scale_factor=upsample_time, mode='trilinear')
                space = space.permute(0, 2, 3, 4, 1)
                
            for k in mov_data.keys():
                mov_data[k] = DeformerZoo.get_deformer_by_constrain(k)(fix_data[k], mov_data[k], space)

            spaces[i] = space
        
        # crop the data due to the incompleteness of the moving image.
        # for k in mov_data.keys():
        #     for space in spaces:
        #         space = torch.Tensor(space)
        #         space = space.permute(0, 4, 1, 2, 3)
        #         mov_data[k]["img_raw"][space[:, 0:1, ...]<=-1] = 0
        #         mov_data[k]["img_raw"][space[:, 0:1, ...]>=1] = 0
        #         mov_data[k]["img_raw"][space[:, 1:2, ...]<=-1] = 0
        #         mov_data[k]["img_raw"][space[:, 1:2, ...]>=1] = 0
        #         mov_data[k]["img_raw"][space[:, 2:3, ...]<=-1] = 0
        #         mov_data[k]["img_raw"][space[:, 2:3, ...]>=1] = 0
        return mov_data

    @staticmethod
    def put_to_cuda(data: dict):
        for k in data.keys():
            for sub_k in data[k].keys():
                if isinstance(data[k][sub_k], torch.Tensor):
                    data[k][sub_k] = data[k][sub_k].cuda()
        return data

    @staticmethod
    def put_to_cpu(data: dict):
        for k in data.keys():
            for sub_k in data[k].keys():
                if isinstance(data[k][sub_k], torch.Tensor):
                    data[k][sub_k] = data[k][sub_k].cpu()
        return data
