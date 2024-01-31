
"""
Generate a data json (train.json, val.json adn test.json)
data folders should be appointed below
moving path can be more than one
fix path: the fixed image data folder (e.g. allen)
"""

import random
import numpy as np
import json
from random import shuffle
import os


random.seed(3)

# specify your data here
mov_dir = "../../data/data_4/moving/"
mov_path_list = [os.path.join(mov_dir, i) for i in os.listdir(mov_dir) if os.path.isdir(os.path.join(mov_dir, i))]

# specify the target data here
fix_path = "../../data/data_4/fix/allen"

# specify the output folder data here
dst = "../../data/jsons/data_4"
#

if not os.path.exists(dst):
    os.makedirs(dst)

data_train = {"moving": [], "fix": []}
data_test = {"moving": [], "fix": []}
data_tot = {"moving": [], "fix": []}


shuffle(mov_path_list)
for i in range(int(len(mov_path_list)/5*4)):
    data_train["moving"].append(mov_path_list[i])
    data_train["fix"].append(fix_path)

for i in range(len(mov_path_list)):
    data_tot["moving"].append(mov_path_list[i])
    data_tot["fix"].append(fix_path)

for i in range(int(len(mov_path_list)/5*4), len(mov_path_list)):
    data_test["moving"].append(mov_path_list[i])
    data_test["fix"].append(fix_path)

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