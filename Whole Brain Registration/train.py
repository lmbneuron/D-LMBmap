import os

from shutil import copyfile
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast
from ms_regnet.constrain.loss import fusion_loss
from visparam import *
from optimizer import Optimizer
from ms_regnet.core import *
from ms_regnet.tools import ModelSaver, set_random_seed
from ms_regnet.tools import Augment
from base import Baser


class Trainer(Baser):
    def __init__(self,
                 config,
                 basedir,
                 config_path):
        """
        :param config: context of the config file
        :param basedir: the output dir to save the result
        :param config_path: path of the config file
        """
        super().__init__()
        self.config = config
        self.basedir = basedir
        copyfile(config_path, os.path.join(self.basedir, "config.yaml"))
        self.gpu_num = len(self.config["TrainConfig"]["gpu"].split(","))
        self.config["TrainConfig"]["batch"] *= self.gpu_num
        self.train_type = self.config["TrainConfig"].get("train_type", 0)

        set_random_seed()
        print(f"base dir is {self.basedir}")
        self.writer = SummaryWriter(log_dir=os.path.join(self.basedir, "logs"))
        self.model_saver = ModelSaver(self.config["TrainConfig"].get("max_save_num", 10))
        self.train_loader = self._get_loader(os.path.join(self.config["TrainConfig"]["data"], "train.json"),
                                             self.config, shuffle=True)
        self.val_loader = self._get_loader(os.path.join(self.config["TrainConfig"]["data"], "test.json"),
                                           self.config, shuffle=False)
        self.model = self._get_model(self.config)
        self.scaler = GradScaler()
        self.mp_train = self.config["TrainConfig"]["mixed_precision_train"]
        self.augment = Augment(self.config["DataConfig"]["use_deform"], self.config["DataConfig"]["use_crop"])
        self.optimizer = Optimizer(self.config["OptimConfig"], self.model,
                                   checkpoint=self.config["TrainConfig"]["checkpoint"])

        self.optimizer.zero_grad()
        if self.gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model, dim=0)

    def train(self):
        self.step = 0
        for self.epoch in range(self.config["TrainConfig"]["epoch"]):
            print(f"epoch: {self.epoch}")
            self.model.train()
            self.train_loss = {}
            self.train_metric = {}
            for input in tqdm(self.train_loader):
                input = self.augment(input)
                fix_name = input["fix"]["name"]
                mov_name = input["mov"]["name"]
                torch.cuda.empty_cache()
                self._train(input)
                del input
                torch.cuda.empty_cache()

            # print the reslut
            print(f"epoch: {self.epoch} train loss: {np.mean(self.train_loss['tot_loss'])} lr {self.optimizer.get_cur_lr()}")
            for k, v in self.train_metric.items():
                print(f"metric: {k}: {np.mean(v)}")
            for k, v in self.optimizer.get_cur_lr().items():
                self.writer.add_scalar("lr/"+k, v, self.step)
            for k, v in self.train_loss.items():
                self.writer.add_scalar("train/loss/"+k, np.mean(v), self.step)
            for k, v in self.train_metric.items():
                self.writer.add_scalar("train/metric/"+k, np.mean(v), self.step)

            # testing result
            if self.epoch % 100 == 99:
                self.val_loss = {}
                self.val_metric = {}
                self.model.eval()
                for input in tqdm(self.val_loader):
                    torch.cuda.empty_cache()
                    self._eval(input)
                    del input
                    torch.cuda.empty_cache()
                # print the result
                print(f"epoch: {self.epoch} eval loss: {np.mean(self.val_loss['tot_loss'])}")
                for k, v in self.val_metric.items():
                    print(f"eval metric: {k}: {np.mean(v)}")
                for k, v in self.val_loss.items():
                    self.writer.add_scalar("eval/loss/" + k, np.mean(v), self.step)
                for k, v in self.val_metric.items():
                    self.writer.add_scalar("eval/metric/" + k, np.mean(v), self.step)

    def _train(self, input):
        fix_name = input["fix"]["name"]
        mov_name = input["mov"]["name"]
        # print(mov_name)
        input["fix"].pop("name")
        input["mov"].pop("name")
        input = tensor_cuda(input)
        if self.mp_train:
            input = tensor_half(input)

        self.optimizer.zero_grad()
        if self.mp_train:
            ## Mixed precision training
            with autocast():
                output = self._train_forward(input)
            self.scaler.scale(output["loss"]["tot_loss"]).backward()
            for k, optim in self.optimizer.optimizer_dict.items():
                if self.config["ModelConfig"]["scale"] > 1 and ("rigid" in k or "affine" in k):
                    continue
                self.scaler.step(optim)
            for lr_scheduer in self.optimizer.lr_scheduler_dict.values():
                lr_scheduer.step()
            self.scaler.update()
        else:
            output = self._train_forward(input)
            output["loss"]["tot_loss"].backward()
            self.optimizer.step()
        update_dict(self.train_loss, output["loss"])
        if self.step % 200 == 199:
            for k in input["fix"].keys():
                visual_img("train/"+k, fix_name[0]+"_"+mov_name[0], input["fix"][k]["img_raw"][0][0], input["mov"][k]["img_raw"][0][0],
                        [i["reg"][k]["img_raw"][0][0] for i in output["reg"]], self.writer, self.step)
            visual_gradient(self.model, self.writer, self.step)
            self.model_saver.save(os.path.join(self.basedir, "checkpoint", str(self.step).zfill(4)+".pth"),
                                    {"model": self.model.state_dict(), "optim": self.optimizer.state_dict()})
        torch.cuda.empty_cache()
        self.step += 1

    def _eval(self, input):
        fix_name = input["fix"]["name"]
        mov_name = input["mov"]["name"]
        input["fix"].pop("name")
        input["mov"].pop("name")
        input = tensor_cuda(input)
        with torch.no_grad():
            output = self._eval_forward(input)
            update_dict(self.val_loss, output["loss"])
        for k in input["fix"].keys():
            visual_img("eval/" + k, fix_name[0]+"_"+mov_name[0], input["fix"][k]["img_raw"][0][0], input["mov"][k]["img_raw"][0][0],
                       [i["reg"][k]["img_raw"][0][0] for i in output["reg"]], self.writer, self.step)
        del output
        torch.cuda.empty_cache()

    def _train_forward(self, input):
        return self._forward(input, self.train_metric)

    def _eval_forward(self, input):
        return self._forward(input, self.val_metric)

    def _forward(self, input, metric):
        output = self.model(input)
        output = self._ave_loss(output)
        update_dict(metric, self.cal_metric(output["reg"][-1]["reg"], input["fix"], self.config.get("MetricConfig")))
        loss = fusion_loss(output["loss"], self.config["LossConfig"], self.config["DataConfig"]["constrain"])
        output["loss"]["tot_loss"] = loss
        return output

    def _ave_loss(self, output):
        """
        calculate the average of the network output losses
        :param output:
        :return:
        """
        for k in output["loss"].keys():
            output["loss"][k] = torch.mean(output["loss"][k])
        return output


def update_dict(tot_dict: dict, cur_dict: dict):
    """
    update the element of dictionary
    """
    for k in cur_dict.keys():
        if tot_dict.get(k) is None:
            if isinstance(cur_dict[k], torch.Tensor):
                tot_dict[k] = [cur_dict[k].cpu().detach().numpy()]
            else:
                tot_dict[k] = [cur_dict[k]]
        else:
            if isinstance(cur_dict[k], torch.Tensor):
                tot_dict[k].append(cur_dict[k].cpu().detach().numpy())
            else:
                tot_dict[k].append(cur_dict[k])


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()