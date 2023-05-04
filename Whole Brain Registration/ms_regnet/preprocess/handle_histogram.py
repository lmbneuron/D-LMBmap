import cv2
import sys
import matplotlib.pyplot as plt

from ..tools.io import read_tiff_stack, write_tiff_stack, read_nii
import numpy as np


def main():
    path = r"E:\soma\181208_488_12-03-52\181208_488_12-03-52.tiff"
    handle_intensity2d_by_filename(path)


def handle_intensity2d_by_filename(path):
    import os
    mydata = read_tiff_stack(path)
    mydata = handle_intensity3d(mydata)
    write_tiff_stack(mydata, os.path.join(os.path.split(path)[0], "ahe" + os.path.split(path)[1]))
    return os.path.join(os.path.split(path)[0], "ahe" + os.path.split(path)[1])


def handle_intensity2d(volume):
    print(np.shape(volume))
    for i in range(volume.shape[0]):
        img = volume[i]
        img_eq = adaptive_history_equal(img)
        volume[i] = img_eq
    return volume


def handle_intensity3d(volume):
    H, W, C = volume.shape
    stack = [i for i in volume]  ## H个 W*C
    stack = np.concatenate(stack, axis=0) ## H*W, C
    # stack = cv2.equalizeHist(stack)
    stack = adaptive_history_equal(stack)
    volume = [stack[i*W: (i+1)*W] for i in range(H)]
    volume = np.reshape(volume, newshape=(H, W, C))
    return volume


def history_equal(image):
    img_eq = cv2.equalizeHist(image)
    return img_eq


def adaptive_history_equal(image):
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(8, 8))
    img_eq = clahe.apply(image)
    return img_eq


if __name__ == "__main__":
    main()
