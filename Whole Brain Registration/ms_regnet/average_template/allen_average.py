"""
base on the allen average brain template
"""


import os
import warnings

from tqdm import tqdm
import torch
import logging
import numpy as np
from copy import deepcopy
from torch.nn.functional import grid_sample

from ms_regnet.tools import set_random_seed
from ms_regnet.tools.deform import generate_base_deform_space, flowinverse
from .fusion import FusionTool
from .inverter import Inverter
from base import Baser
from ms_regnet.datasets import SimpleBrain


class Averager(Baser):
    def __init__(self,
                 config,
                 basedir,
                 ckp_path,
                 cfg_path):
        super(Averager, self).__init__()
        self.config = config
        self.basedir = basedir
        self.ckp_path = ckp_path
        self.cfg_path = cfg_path
        self.fusion_tool = None
        self.inverter = Inverter(5)
        self.model = self._get_model(self.config,
                                     no_loss=True,
                                     no_space=False,
                                     checkpoint=self.ckp_path,
                                     eval=True)

        print(f"base dir is {self.basedir}")

    def average(self):
        set_random_seed(self.config["TrainConfig"].get("seed", 0))
        loader = self._get_loader(os.path.join(self.config["TrainConfig"]["data"], "tot.json"),
                                  self.config,
                                  shuffle=True,
                                  batch_size=1
                                  )
        logging.basicConfig(filename=os.path.join(self.basedir, "log.txt"), filemode='w', level=logging.INFO)
        final_template = self._select_initial_template("/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/result/bezier_atlas_3/template/", 
                                                        self.config["DataConfig"]["constrain"])
        step = 3
        while True:
            step += 1
            print(f"step: {step}")
            self.fusion_tool = FusionTool()
            cur_space = None
            space_num = 0
            for data in tqdm(loader):
                data = data["mov"]
                name = data["name"][0]
                print(f"[{__file__}]: reading data: {name}")
                data.pop("name")

                reg_data, spaces = self._inference(final_template, data)
                torch.cuda.empty_cache()
                self.write_data(reg_data, name, self.basedir, str(step))
                import pickle
                with open(os.path.join(self.basedir, str(step), name, "space.pkl"), "wb") as f:
                    pickle.dump(spaces, f)

                # calculate the average registration result by pixels
                self.fusion_tool.fusion(reg_data)

                space = self.combine_space(spaces)
                if cur_space is None:
                    cur_space = space
                else:
                    cur_space += space
                space_num += 1

            cur_template = self.fusion_tool.finish_fusion()
            cur_space /= space_num
            base_space = generate_base_deform_space((1, 1, cur_space.shape[1], cur_space.shape[2], cur_space.shape[3]),
                                                    'cpu').numpy()
            loss = np.mean(np.abs(cur_space-base_space))
            print(f"loss: {loss}")

            # invert
            # final_mov = self.inverter.invert(final_template, cur_space)
            final_template, space = self.inverter.invert(cur_template, cur_space)
            import pickle
            with open(os.path.join(self.basedir, str(step), "space.pkl"), "wb") as f:
                pickle.dump(space, f)
            self.write_data(final_template, "template", self.basedir, str(step))
            # write_data(final_mov, str(step) + "_mov", self.basedir)
            # final_template = final_mov
            break

    def _select_initial_template(self, filepath, constraints):
        dataset = SimpleBrain([filepath], constraints)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        for data in loader:
            name = data["name"]
            data.pop("name")
            return data

    def _inference(self, fix, mov):
        data_d = deepcopy(mov)
        template_d = deepcopy(fix)

        data_d = self._downsample(data_d)
        template_d = self._downsample(template_d)

        template_d = self.put_to_cuda(template_d)
        data_d = self.put_to_cuda(data_d)

        net_input = {"fix": template_d, "mov": data_d}
        with torch.no_grad():
            output = self.model(net_input)
            spaces = [i["space"] for i in output["reg"]]
            reg_data = self.deform(fix, mov, spaces, need_upsample=True)
            reg_data = self.put_to_cpu(reg_data)
        return reg_data, spaces

    def _downsample(self, data: dict):
        new_data = deepcopy(data)
        for k in new_data.keys():
            new_data[k]["img"] = new_data[k]["img"][:, :, ::2, ::2, ::2]
            new_data[k]["img_raw"] = new_data[k]["img_raw"][:, :, ::2, ::2, ::2]
        return new_data

    def combine_space(self, spaces):
        """
        Be careful!!!
        This function may causes the distance of the
        corresponding points between fix image and moving image is so far.
        In another word, the combined space is not the real combination of the spaces.
        """
        cur = spaces[0]
        cur = np.transpose(cur, (0, 4, 1, 2, 3))
        cur = torch.Tensor(cur)
        for i in range(1, len(spaces)):
            cur = grid_sample(cur, torch.Tensor(spaces[i]), mode='nearest', align_corners=True)
        cur = cur.permute(0, 2, 3, 4, 1)
        return cur.numpy()



