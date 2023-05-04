"""
extract outline base on raw image
"""

import numpy as np
import cv2
import sys

sys.path.append("../../")
sys.setrecursionlimit(100000)

from ms_regnet.tools.io import read_tiff_stack, write_tiff_stack, read_mhd

# threshold = 40 ##threshold that judge a pixel inside or outside brain
threshold = 20


def main():
    from glob import glob

    path1_list = glob(r"/media/data1/zht/Recursive_network_pytorch/2021-05-18_16-14-09/*/*hole.tiff")
    path2_list = glob(r"/media/data1/zht/Recursive_network_pytorch/rigid_part_data/data_4/fix/*/*convex.tiff")
    path1_list.sort()
    path2_list.sort()
    print(path1_list)
    print(path2_list)
    for path1, path2 in zip(path1_list, path2_list):
        print(path1)
        evaluate_by_iou(path1, path2)


def evaluate_by_iou(path1,
                    path2):
    if path1.endswith("mhd"):
        vol1 = read_mhd(path1)
    elif path1.endswith("tiff"):
        vol1 = read_tiff_stack(path1)
    else:
        raise AttributeError

    if path2.endswith("mhd"):
        vol2 = read_mhd(path2)
    elif path2.endswith("tiff"):
        vol2 = read_tiff_stack(path2)
    else:
        raise AttributeError
    vol2 = vol2.astype(np.uint8)
    iou = cal_iou(vol1, vol2)
    print(vol1.shape)
    print(vol2.shape)
    print(iou)


def get_mask(path, tarpath, thres=20, label=255):
    global threshold
    threshold = thres
    if path.endswith("mhd"):
        vol = read_mhd(path)
    elif path.endswith("tiff"):
        vol = read_tiff_stack(path)
    else:
        raise AttributeError
    mask = segment(vol)
    mask[mask > 0] = label
    write_tiff_stack(mask, tarpath)


def get_start_point_list(img):
    """
    start from four corners
    """
    start_list = []
    for i in range(img.shape[1]):
        start_list.append((0, i))
        start_list.append((img.shape[0] - 1, i))
    for i in range(img.shape[0]):
        start_list.append((i, 0))
        start_list.append((i, img.shape[1] - 1))
    return start_list


def get_neibor(cur, img):
    displacement = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    neibor_list = []
    for delta in displacement:
        if 0 <= cur[0] + delta[0] < img.shape[0] \
                and 0 <= cur[1] + delta[1] < img.shape[1]:
            neibor_list.append((cur[0] + delta[0], cur[1] + delta[1]))
    return neibor_list


def is_in_brain(cur, img):
    if img[cur[0], cur[1]] >= threshold:
        return True
    else:
        return False


def bfs(img):
    queue = []
    visited = np.zeros(shape=img.shape, dtype=np.bool)
    start_list = get_start_point_list(img)
    for st in start_list:
        queue.append(st)
    while len(queue) > 0:
        cur = queue[0]
        queue.pop(0)
        if visited[cur[0], cur[1]]:
            continue
        if is_in_brain(cur, img):
            continue
        visited[cur[0], cur[1]] = True
        neighbor_list = get_neibor(cur, img)
        for neighbor in neighbor_list:
            queue.append(neighbor)
    visited = 1 - visited
    return visited


def segment(vol):
    mask = np.zeros(shape=vol.shape, dtype=np.uint8)
    for i, img in enumerate(vol):
        mask[i] = bfs(img)
    return mask


def cal_iou(vol1, vol2):
    """
    calculate iou of two image
    pixel less than 20(uint8 image) are the background, else are brain
    """
    if vol1.shape != vol2.shape:
        vol1 = vol_resize(vol1, dsize=(vol2.shape[1], vol2.shape[2]))
    vol1[vol1 <= 20] = 0
    vol1[vol1 > 20] = 1
    vol2[vol2 <= 20] = 0
    vol2[vol2 > 20] = 1
    vol1 = vol1.astype(np.uint8)
    vol2 = vol2.astype(np.uint8)
    vol_i = vol2.copy()
    vol_i[vol1 == 0] = 0
    vol_i[vol_i != 0] = 1
    vol_u = vol2.copy()
    vol_u[vol1 != 0] = 1
    vol_u[vol_u != 0] = 1
    i = np.sum(vol_i)
    u = np.sum(vol_u)
    return i / u


def cal_iin1(vol1, vol2):
    if vol1.shape != vol2.shape:
        vol1 = vol_resize(vol1, dsize=(vol2.shape[1], vol2.shape[2]))
    vol_i = vol2.copy()
    vol_i[vol1 == 0] = 0
    vol_i[vol_i != 0] = 1
    vol_1 = vol1.copy()
    vol_1[vol_1 != 0] = 1
    i = np.sum(vol_i)
    sum1 = np.sum(vol_1)
    return i / sum1


def vol_resize(vol, dsize):
    newvol = []
    for img in vol:
        img = cv2.resize(img, dsize=(dsize[1], dsize[0]), interpolation=cv2.INTER_NEAREST)
        newvol.append(img)
    newvol = np.reshape(newvol, newshape=(-1, dsize[0], dsize[1]))
    return newvol


if __name__ == "__main__":
    # path = r"D:\soma\181208_15_21_38\elastix_hpf\2\result.mhd"
    # vol1 = read_mhd(path)
    # vol1 = vol1.astype(np.uint8)
    # write_tiff_stack(vol1, path.replace("mhd", "tiff"))
    evaluate_by_iou(r"D:\soma\181208_15_21_38\181208_15_21_38_outline.tiff",
                    r"D:\soma\181208_15_21_38\elastix_outline\2\result.mhd")
