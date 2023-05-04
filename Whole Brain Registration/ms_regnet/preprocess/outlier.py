import SimpleITK as sitk
import numpy as np
import os
import cc3d
import tqdm
import cv2

def thined_component(stack, vol):
    labels = cc3d.connected_components(stack)
    for i in tqdm.tqdm(range(1, labels.max() + 1)):
        label = labels == i
        # print(label.astype(np.int).sum())
        if label.astype(np.int).sum() < vol:
            labels[label] = 0
    labels[labels > 0] = 1
    return labels

def get_num_count(label, num):
    mask = (label == num)
    arr_new = label[mask]
    return arr_new.size


if __name__ == '__main__':
    import sys
    sys.path.append("../..")
    from ms_regnet.tools.io import read_tiff_stack, write_tiff_stack
    from glob import glob

    path_list = glob(r"D:\mh\zsl8\*hpf.tiff")
    for path in path_list:
        print(path)
        opt_nii = read_tiff_stack(path)
        outline = read_tiff_stack(path.replace("hpf.tiff", "outline.tiff"))
        opt_nii[outline==0] = 0

        # # process outlier
        # opt_nii = opt_nii[::-1, ...]
        # opt_nii[297:, :, :] = 0
        # opt_nii[244:, :, 484:] = 0
        # # opt_nii[:11, :, :] = 0
        # # opt_nii[294:, :, :] = 0
        # opt_nii = opt_nii[::-1, ...]
        label = np.max(opt_nii)

        opt_nii = thined_component(opt_nii, 1000000)         #100000CP、V
        opt_nii[opt_nii>0] = label
        opt_nii = opt_nii.astype(np.uint8)

        write_tiff_stack(opt_nii, path.replace("hpf", "hpf2"))


