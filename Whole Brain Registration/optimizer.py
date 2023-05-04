import torch

import lr_scheduler


class Optimizer:
    def __init__(self, config, model, checkpoint=None):
        self.optimizer_dict = {}
        self.lr_scheduler_dict = {}
        for net_type in model.net.keys():
            if len(model.net[net_type].state_dict()) > 0:
                self.optimizer_dict[net_type] = self._get_optim(config[net_type]["optimizer"]["type"],
                                                                config[net_type]["optimizer"]["params"],
                                                                model.net[net_type])

        for net_type in model.net.keys():
            if len(model.net[net_type].state_dict()) > 0:
                self.lr_scheduler_dict[net_type] = self._get_lr_scheduler(config[net_type]["lr_scheduler"]["type"],
                                                                config[net_type]["lr_scheduler"]["params"],
                                                                self.optimizer_dict[net_type])
        if checkpoint is not None and config["load_checkpoint"]:
            self._load(checkpoint)

    def _load(self, checkpoint):
        state_dict = torch.load(checkpoint)["optim"]
        for k in self.optimizer_dict.keys():
            self.optimizer_dict[k].load_state_dict(state_dict["optim"][k])
        for k in self.lr_scheduler_dict.keys():
            self.lr_scheduler_dict[k].load_state_dict(state_dict["lr_scheduler"][k])

    def step(self):
        for v in self.optimizer_dict.values():
            v.step()
        for v in self.lr_scheduler_dict.values():
            v.step()

    def zero_grad(self):
        for v in self.optimizer_dict.values():
            v.zero_grad()

    def get_cur_lr(self):
        lr_dict = {}
        for type, v in self.lr_scheduler_dict.items():
            lr_dict[type] = v.get_last_lr()[0]
        return lr_dict

    def _get_optim(self, type: str, params: dict, model):
        """
        :param type: classifier
        :param params: parameters included
        :param model: Model that this optimizer is used
        :return:
        """
        return getattr(torch.optim, type, None)(params=model.parameters(), **params)

    def _get_lr_scheduler(self, type: str, params: dict, optim):
        """
        :param type: lr_scheduler type
        :param params: parameters included这个lr_scheduler含有的参数列表
        :param optim: optimizer that this optmizer is used
        :return:
        """
        return lr_scheduler.get_instance(type, params, optim)

    def state_dict(self):
        return {
                "optim": {k: v.state_dict() for k, v in self.optimizer_dict.items()},
                "lr_scheduler": {k: v.state_dict() for k, v in self.lr_scheduler_dict.items()}
                }
