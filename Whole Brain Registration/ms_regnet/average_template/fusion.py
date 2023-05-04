"""
method of fusing two brain into one
"""

import numpy as np
import torch

import tqdm

from ms_regnet.constrain.reader_zoo import restore_raw_image_from_output


class FusionTool:
    def __init__(self):
        self.mask = {}
        self.result = {}
        self.number = 0

    def fusion(self, reg: dict):
        self.number += 1

        for k in reg.keys():
            if self.number == 1:
                self.result[k] = {}
                if k == "simi":
                    self.result[k]["img_raw"] = [restore_raw_image_from_output(reg[k]).astype(np.float32)]
                else:
                    self.result[k]["img_raw"] = reg[k]["img_raw"].clone()
                self.result[k]["img"] = reg[k]["img"].clone()
            else:
                if k == 'simi':
                    self.result[k]["img_raw"].append(restore_raw_image_from_output(reg[k]).astype(np.float32))
                else:
                    self.result[k]["img_raw"] += reg[k]["img_raw"]
                self.result[k]["img"] += reg[k]["img"]

            if k != 'simi':
                for sub_k in reg[k].keys():
                    if sub_k not in self.result[k]:
                        self.result[k][sub_k] = reg[k][sub_k]

    def finish_fusion(self):
        import pickle
        with open("data.pkl", "wb") as f:
            pickle.dump(self.result, f)
        for k in self.result.keys():
            if k == 'simi':
                vols = self.result[k]["img_raw"]
                raw_vol = np.zeros_like(vols[0], dtype=np.float32)
                print(f"[{__file__}]: finishing fusion")
                for x in tqdm.tqdm(range(vols[0].shape[0])):
                    for y in range(vols[0].shape[1]):
                        for z in range(vols[0].shape[2]):
                            datas = [i[x, y, z] for i in vols]
                            raw_vol[x, y, z] = sum(datas) / len(datas)

                self.result[k]["img_raw"] = raw_vol[np.newaxis, np.newaxis, ...]
                self.result[k]["img_raw"] = torch.tensor(self.result[k]["img_raw"], dtype=self.result[k]["img"].dtype,
                                                    device=self.result[k]["img"].device)
                self.result[k]["min"] = torch.min(self.result[k]["img_raw"])
                self.result[k]["max"] = torch.max(self.result[k]["img_raw"])
                self.result[k]["img_raw"] = (self.result[k]["img_raw"] - self.result[k]["min"]) / (self.result[k]["max"] - self.result[k]["min"])
            else:
                self.result[k]["img_raw"] /= self.number
            self.result[k]["img"] /= self.number
        return self.result
