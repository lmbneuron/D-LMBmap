from tqdm import tqdm
import numpy as np
import os
import torch
from shutil import copyfile
import logging

from base import Baser
from ms_regnet.core import *

NO_SPACE = False  # if saving a deform space


class Inference(Baser):
    def __init__(self,
                 config,
                 basedir,
                 ckp_path,
                 cfg_path):
        super().__init__()
        self.config = config
        self.basedir = basedir
        self.ckp_path = ckp_path
        self.cfg_path = cfg_path
        self.train_type = self.config["TrainConfig"].get("train_type", 0)
        print(f"base dir is {self.basedir}")

    def inference(self):
        logging.basicConfig(filename=os.path.join(self.basedir, "log.txt"), filemode='w', level=logging.INFO)
        copyfile(self.ckp_path, os.path.join(self.basedir, "checkpoint", "checkpoint.pth"))
        copyfile(self.cfg_path, os.path.join(self.basedir, "config.yaml"))
        self.loader = self._get_loader(os.path.join(self.config["TrainConfig"]["data"], "tot.json"),
                                self.config, shuffle=False, batch_size=1)

        self.model = self._get_model(self.config, no_loss=True, no_space=NO_SPACE, checkpoint=self.ckp_path)
        # self.augment = Augment(self.config["DataConfig"]["use_deform"], self.config["DataConfig"]["use_crop"])
        self.model.eval()
        for input in tqdm(self.loader):
            torch.cuda.empty_cache()
            self._inference(input)
            del input
            torch.cuda.empty_cache()

    def _inference(self, input):
        fix_name = input["fix"]["name"]
        mov_name = input["mov"]["name"]
        input["fix"].pop("name")
        input["mov"].pop("name")
        with torch.no_grad():
            input = tensor_cuda(input)
            output = self.model(input)
            logging.info(fix_name + mov_name)
            metric_dict = self.cal_metric(output["reg"][-1]["reg"], input["fix"], self.config.get("MetricConfig"))
            for k, v in metric_dict.items():
                logging.info(f"{k} {v}")

            folder_name = fix_name[0] + '_' + mov_name[0]
            # save the fix and moving img after registration

            self.write_data(input["fix"], name=folder_name, basedir=self.basedir, parent_name="fix")
            self.write_data(input["mov"], name=folder_name, basedir=self.basedir, parent_name="mov")
            self.write_data(output["reg"][-1]["reg"], name=folder_name, basedir=self.basedir, parent_name="reg")

            # save the deform space
            for i in range(len(output["reg"])):
                if output["reg"][i].get("space") is not None:
                    np.save(os.path.join(self.basedir, "reg", folder_name, str(i) + "_space"), output["reg"][i]["space"])
        del output
        del metric_dict
        torch.cuda.empty_cache()


if __name__ == "__main__":
    infer = Inference()
    infer.inference()
