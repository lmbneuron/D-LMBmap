from torch import nn, load
from copy import deepcopy
import torch
from .backbone import VTNAffineModule, create_backone, Rigid


class CascadedNet(nn.Module):
    def __init__(self,
                 rigid_config: dict,
                 affine_config: dict,
                 backbone_config: dict,
                 loss,
                 scale: int,
                 checkpoint: str,
                 no_loss: bool = False,
                 no_space: bool = True):
        super(CascadedNet, self).__init__()
        backbone = create_backone(backbone_config["type"])
        self.net = nn.ModuleDict()
        self.net["rigid"] = nn.ModuleList()
        for i in range(rigid_config["num"]):
            self.net["rigid"].append(Rigid(loss, scale, rigid_config, no_loss=no_loss, no_space=no_space))
        self.net["affine"] = nn.ModuleList()
        for i in range(affine_config["num"]):
            self.net["affine"].append(VTNAffineModule(loss, scale, affine_config, no_loss=no_loss, no_space=no_space))
        self.net["backbone"] = nn.ModuleList()
        for i in range(backbone_config["num"]):
            self.net["backbone"].append(backbone(loss,
                                                 scale,
                                                 backbone_config,
                                                 no_loss=no_loss,
                                                 no_space=no_space))
        if len(checkpoint) > 0:
            self._load(checkpoint)

    def _load(self, checkpoint):
        cur_dict = self.state_dict()
        if torch.cuda.is_available():
            need_dict = load(checkpoint)["model"]
        else:
            need_dict = load(checkpoint, map_location='cpu')["model"]
        print(need_dict.keys())
        print("the network load from the ", checkpoint)
        for ck, cv in cur_dict.items():
            if need_dict.get("module." + ck) is not None:
                print(ck, " has been loader from ", checkpoint)
                cur_dict[ck] = need_dict["module." + ck]
            elif need_dict.get(ck) is not None:
                print(ck, " has been loader from ", checkpoint)
                cur_dict[ck] = need_dict[ck]
            else:
                print(ck, " random init")
        self.load_state_dict(cur_dict)

    def forward(self, input_dict):
        """
        :param: input_dict: {
                    fix: {
                        name:  brain name of fixed image
                        simi (processed fixed image）：{
                            img: image data
                        }
                        outline (mask of image outline): {
                            img: data of outline
                        }
                        hole (mask of image ventricle) : {
                            img: data of ventricle
                        }
                        ...
                    }
                    mov: {
                        name:  brain name of moving image
                        simi (processed moving image）：{
                            img: image data
                        }
                        outline (marked data of image outline): {
                            img: data of outline
                        }
                        hole (marked data of image ventricle) : {
                            img: data of ventricle
                        }
                        ...
                    }
                }
        :return: { "fix": same as input，
                   "mov": same as input，
                   "loss": {}，
                   "reg": { [] }
                  }
        """
        fix_img, mov_img = input_dict["fix"].copy(), input_dict["mov"].copy()
        input_dict = deepcopy(input_dict)
        loss = {}
        result_dict = {"reg": []}

        def _register(net: nn.Module):
            one_result_dict = net(input_dict)
            result_dict["reg"].append(one_result_dict)
            '''result of current round will be next round's moving image'''
            for k in input_dict["mov"].keys():
                for subk in input_dict["mov"][k].keys():
                    input_dict["mov"][k][subk] = one_result_dict["reg"][k][subk].detach()
            self._update_loss(loss, one_result_dict["loss"])
            torch.cuda.empty_cache()

        for k, module_list in self.net.items():
            for module in module_list:
                _register(module)

        torch.cuda.empty_cache()
        # calculate average loss from rigid,affine and voxel morph module
        for key in loss.keys():
            loss[key] = loss[key].mean()
        result_dict = {**result_dict,
                       **{
                           "mov": mov_img,
                           "fix": fix_img,
                           "loss": loss,
                       }
                       }
        return result_dict

    def _update_loss(self, tot_loss: dict, cur_loss: dict):
        for k in cur_loss.keys():
            if tot_loss.get(k) is None:
                tot_loss[k] = cur_loss[k]
            else:
                tot_loss[k] = tot_loss[k] + cur_loss[k]
