# accuracy testing

import csv
from tqdm import tqdm
from copy import deepcopy
import os
import torch

from shutil import copyfile
import logging
import yaml

from base import Baser
from ms_regnet.core import *
from ms_regnet.constrain import MetricZoo, DeformerZoo


class Tester(Baser):
    def __init__(self,
                 config,
                 basedir,
                 ckp_path,
                 cfg_path,
                 test_cfg_path,
                 upsample: int):
        super().__init__()
        self.config = config
        self.basedir = basedir
        self.cfg_path = cfg_path
        self.ckp_path = ckp_path
        self.test_cfg_path = test_cfg_path
        self.upsample = upsample
        self.train_type = self.config["TrainConfig"].get("train_type", 0)

        with open(test_cfg_path, "r") as f:
            self.test_cfg = yaml.load(f)

        self.csv_file = open(os.path.join(self.basedir, "result.csv"), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.constrains = [i for i, j in self.test_cfg["DataConfig"]["constrain"].items() if j]
        self.csv_writer.writerow(self.constrains)
        self.test_input = {}

        print(f"base dir is {self.basedir}")

    def test(self):
        logging.basicConfig(filename=os.path.join(self.basedir, "log.txt"), filemode='w', level=logging.INFO)
        copyfile(self.cfg_path, os.path.join(self.basedir, "config.yaml"))
        copyfile(self.test_cfg_path, os.path.join(self.basedir, "test_config.yaml"))

        test_loader = self._get_loader(os.path.join(self.test_cfg["TrainConfig"]["data"], "tot.json"),
                                            self.test_cfg, shuffle=False, batch_size=1)

        self.model = self._get_model(self.config, no_loss=True, no_space=False, checkpoint=self.ckp_path)
        for test_input in tqdm(test_loader):
            self._test(test_input)
        self.csv_file.close()

    def _test(self, test_input):
        fix_name = test_input["fix"]["name"][0]
        mov_name = test_input["mov"]["name"][0]
        test_input["fix"].pop("name")
        test_input["mov"].pop("name")
        folder_name = fix_name + "_" + mov_name
        input = self._get_input_from_config(folder_name)
        if input is None:
            print(f"!!!! wa cannot found test input {folder_name}")
            return
        spaces = []
        input = tensor_cuda(input)
        with torch.no_grad():
            output = self.model(input)
            # save the deform space
            spaces += [output["reg"][i]["space"] for i in range(len(output["reg"]))]
        logging.info(folder_name)
        input = test_input
        if self.upsample > 1:
            reg = self.deform(input["fix"], input["mov"], spaces, need_upsample=True, upsample_time=self.upsample)
        else:
            reg = self.deform(input["fix"], input["mov"], spaces, need_upsample=False)
        metric_dict = self.cal_metric(reg, input["fix"], self.test_cfg.get("MetricConfig"))

        for k, v in metric_dict.items():
            logging.info(f"{k} {v}")

        # write the accuracy result into csv file
        data = [metric_dict.get(k, 0) for k in self.constrains]
        data.append(folder_name)
        self.csv_writer.writerow(data)

        # save the fix and moving img after registration
        self.write_data(input["fix"], name=folder_name, basedir=self.basedir, parent_name="fix")
        self.write_data(input["mov"], name=folder_name, basedir=self.basedir, parent_name="mov")
        self.write_data(reg, name=folder_name, basedir=self.basedir, parent_name="reg")

    def _get_input_from_config(self, folder_name):
        loader = self._get_loader(os.path.join(self.config["TrainConfig"]["data"], "tot.json"),
                                       self.config, shuffle=False, batch_size=1)
        for input in tqdm(loader):
            fix_name = input["fix"]["name"][0]
            mov_name = input["mov"]["name"][0]
            input["fix"].pop("name")
            input["mov"].pop("name")
            if fix_name+"_"+mov_name == folder_name:
                return input
        return None