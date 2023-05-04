import torch
from torch import nn
import sys
import numpy as np
from math import ceil, pi, exp
from tqdm import tqdm
# sys.path.append(r"D:\code\MS-RegNet")
sys.path.append("../..")
from ms_regnet.tools.io import read_tiff_stack, write_tiff_stack


ANNOTATION = read_tiff_stack("/media/root/6701ae9d-6612-4271-8d50-522d7f72528b/zht/Recursive_network_pytorch/allen/template_with_filter_annotation.tiff")
SHAPE = ANNOTATION.shape
import pickle
with open("../../myresources/allen.pkl", "rb") as f:
    allen = pickle.load(f)


class GaussFiler:
    def __init__(self, sigma) -> None:
        """3D gauss fiter

        Args:
            sigma (float)
        """
        kernel_size = 2*ceil(2*sigma) + 1
        half_ksize = kernel_size // 2
        kernel = torch.zeros((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32)
        for i in range(-half_ksize, half_ksize+1):
            for j in range(-half_ksize, half_ksize+1):
                for k in range(-half_ksize, half_ksize+1):
                    kernel[0, 0, i, j, k] = 1 / ((2*pi)**1.5 * sigma ** 3) * exp(-(i**2+j**2+k**2) / (2*sigma**2))
        kernel /= torch.sum(kernel)
        conv = torch.nn.Conv3d(1, 1, kernel_size, 1, padding=half_ksize)
        conv.weight = nn.Parameter(kernel, requires_grad=False)
        conv.requires_grad_ = False
        if torch.cuda.is_available():
            self.conv = conv.cuda()
            
    def __call__(self, vol, thres=0.5) -> np.ndarray:
        """
        3D gauss fiter
        Args:
            vol (np.ndarray): vol.ndim == 3, binary image and must be 0-1
        """
        vol = torch.tensor(vol[np.newaxis, np.newaxis, ...], dtype=torch.float32)
        if torch.cuda.is_available():
            vol = vol.cuda()
        vol = self.conv(vol)
        vol = vol.detach().cpu().numpy()
        vol = np.squeeze(vol)
        vol[vol >= thres] = 1
        vol[vol < thres] = 0
        vol = vol.astype(np.uint8)
        return vol


def dfs_get_regions(region_id, regions, depth=0):
    regions.append(region_id)
    for nex_id in allen["neighbor_children"][region_id]:
        dfs_get_regions(nex_id, regions, depth+1)


def get_mask_by_region_id(region_id):
    mask = np.zeros(SHAPE, dtype=np.uint8)
    for id in allen["all_children"][region_id]:
        mask[ANNOTATION==id] = 1
    return mask

if __name__ == "__main__":
    for sd in [1, 2]:
        for thres in [0.3, 0.5, 0.7]:
            regions = []
            big_regions = ["root"]
            filter = GaussFiler(1)
            mask = np.zeros(SHAPE, dtype=np.uint16)
            for region_name in big_regions:
                dfs_get_regions(allen["name2id"][region_name], regions)
            print(len(regions))

            for region in tqdm(regions):
                new_mask = get_mask_by_region_id(region)
                new_mask = filter(new_mask)
                mask[new_mask > 0] = region
            write_tiff_stack(mask, f"{sd}_{thres}.tiff")

