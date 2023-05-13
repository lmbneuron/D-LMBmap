import os
import numpy as np
import sys
sys.path.append("../../")

from ms_regnet.tools.io import read_tiff_stack, write_tiff_stack
from ms_regnet.tools.upsample import trilinear_interpolate


def main():
    dir = "/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/seronine_transfer/data_4/"
    scaler = 4
    dst_shape = (320//scaler, 456//scaler, 528//scaler)
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith("npy"):
                continue
            print(os.path.join(root, file))
            if file.endswith("tiff"):
                image = read_tiff_stack(os.path.join(root, file))
                if image.shape[0] == dst_shape[0] and image.shape[1] == dst_shape[1] \
                        and image.shape[2] == dst_shape[2]:
                    continue
                if "process" in file:
                    image = trilinear_interpolate(image, dsize=(320 // scaler, 456 // scaler, 528 // scaler))
                else:
                    print(file)
                    image = image[::scaler, ::scaler, ::scaler]
                print(image.shape)
                write_tiff_stack(image, os.path.join(root, file))


def ds_npy(path):
    data = np.load(path)
    new_p = []
    for idx in range(data.shape[0]):
        print(data[idx].shape)
        ds_p = data[idx][13] // 2
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    new_p.append((ds_p[0]+i, ds_p[1]+j, ds_p[2]+k))

    new_p = np.reshape(new_p, newshape=data.shape)
    new_p = new_p.astype(data.dtype)
    return new_p

if __name__ == "__main__":
    main()