import random

import numpy as np
import json
from random import shuffle
import os


random.seed(3)
# REVERSE moving and fix. If True: fix is the allen data, moving is the input data
# If False:fix is the input data, moving is allen data
REVERSE = True

# to generate a data json, data folders should be appoint below
# fix path can be more than one
# moving path: the very fixed image data folder
fix_dir_list = ["/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/atlas_test/data/fix/",
                ]
fix_path_list = []
for fix_dir in fix_dir_list:
    fix_path_list += [os.path.join(fix_dir, i) for i in os.listdir(fix_dir) if os.path.isdir(os.path.join(fix_dir, i))]
moving_path = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/atlas_test/data/moving/allen"
dst = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/atlas_test_rev/data/"  # 指定json文件的输出路径
#

if not os.path.exists(dst):
    os.makedirs(dst)

data_train = {"moving": [], "fix": []}
data_test = {"moving": [], "fix": []}
data_tot = {"moving": [], "fix": []}


shuffle(fix_path_list)
for i in range(int(len(fix_path_list)/5*4)):
    data_train["moving"].append(moving_path)
    data_train["fix"].append(fix_path_list[i])

for i in range(len(fix_path_list)):
    data_tot["moving"].append(moving_path)
    data_tot["fix"].append(fix_path_list[i])

for i in range(int(len(fix_path_list)/5*4), len(fix_path_list)):
    data_test["fix"].append(fix_path_list[i])
    data_test["moving"].append(moving_path)

if REVERSE:
    data_train["moving"], data_train["fix"] = data_train["fix"], data_train["moving"]
    data_tot["moving"], data_tot["fix"] = data_tot["fix"], data_tot["moving"]
    data_test["moving"], data_test["fix"] = data_test["fix"], data_test["moving"]

print("train json: len: ", len(data_train["fix"]))
print(data_train)
print("test json: len: ", len(data_test["fix"]))
print(data_test)
with open(os.path.join(dst, "train.json"), "w") as f:
    json.dump(data_train, f, indent=4)

with open(os.path.join(dst, "test.json"), "w") as f:
    json.dump(data_test, f, indent=4)

with open(os.path.join(dst, "tot.json"), "w") as f:
    json.dump(data_tot, f, indent=4)