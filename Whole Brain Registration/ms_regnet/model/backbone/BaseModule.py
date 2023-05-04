from torch import nn
import torch
from typing import Dict
from ...constrain import DeformerZoo, LossZoo
from .fusion import FusionFactory


def conv3d_with_leakyReLU(*args):
    return nn.Sequential(nn.Conv3d(*args),
                         nn.LeakyReLU())


def median_blur(x, kernel_size=5):
    conv = nn.Conv3d(3, 3, kernel_size, padding=(kernel_size - 1) // 2, bias=False, groups=3)
    conv.register_parameter(name='weight',
                            param=nn.Parameter(
                                torch.ones([3, 1, kernel_size, kernel_size, kernel_size]) / (kernel_size ** 3)))
    conv = conv.to(x.device)
    for param in conv.parameters():
        param.requires_grad = False
    x = conv(x)
    return x


class BaseModule(nn.Module):
    def __init__(self,
                 loss: dict,
                 cfg: dict,
                 no_loss: bool = False,
                 no_space: bool = True):
        '''
        :param loss: loss type
        :param cfg: config
        :param no_loss: if loss will be calculate
        :param no_space: if deform space will be saved
        '''
        super(BaseModule, self).__init__()
        self.constrain = cfg["constrain"]
        self.constrain_loss = {}
        self.no_loss = no_loss
        self.no_space = no_space
        self.fusion_module = FusionFactory.create_fusion_module(cfg)
        self.input_channel = self.fusion_module.get_input_channel()
        if not self.no_loss:
            for k, v in self.constrain.items():
                if v:
                    if loss.get(k) is not None:
                        self.constrain_loss[k] = LossZoo.get_loss_by_constrain_and_type(k, loss[k])()
                    else:
                        self.constrain_loss[k] = LossZoo.get_loss_by_constrain(k)()

    def _fix(self):
        """
        Fix the parameters.
        """
        pass

    def forward(self, input_dict):
        pass

    def fusion_input(self, input_dict):
        return self.fusion_module.fusion_input(input_dict)

    def handle_output(self, input_dict, deform_space, result_dict):
        result_dict["reg"] = self.cal_reg(input_dict["fix"], input_dict["mov"], deform_space)
        for k in input_dict["mov"].keys():
            if self.constrain.get(k, False) and self.no_loss is False:
                result_dict["loss"][k] = self.constrain_loss[k](
                    input_dict["fix"][k], input_dict["mov"][k], result_dict["reg"][k], deform_space
                )
            torch.cuda.empty_cache()
        if not self.no_space:
            result_dict["space"] = deform_space.detach().cpu().numpy()
        torch.cuda.empty_cache()

    def cal_reg(self, fix_dict, mov_dict, deform_space) -> Dict:
        reg_dict = {}
        for k in mov_dict.keys():
            reg_dict[k] = DeformerZoo.get_deformer_by_constrain(k)(
                fix_dict.get(k, None),
                mov_dict.get(k),
                deform_space
            )
        return reg_dict



if __name__ == "__main__":
    a = torch.ones((2, 3, 6, 6, 6))
    a = median_blur(a)
