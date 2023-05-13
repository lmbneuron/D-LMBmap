import cv2
import nrrd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append("../..")
from ms_regnet.tools.io import read_tiff_stack, write_tiff_stack


def edge_demo(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edge_output = cv2.Canny(gray, 1, 20)
    return edge_output


graph = []
vol = read_tiff_stack("/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/soma_nuclei/data/moving/mylsfm_bak/mylsfm_annotation.tiff")


def extract_allen_edge():
    for i in tqdm(range(vol.shape[0])):
        slice = vol[i, :, :].astype(float)
        slice /= 1000
        im = np.uint8(plt.cm.jet(slice) * 255)
        graph.append(edge_demo(im))


# 提取allen边缘
extract_allen_edge()
edge = np.array(graph)
write_tiff_stack(edge, r"/media/root/01a3fb01-7912-492d-9561-d36a4da2ffef/zht/data/MS-RegNet/data/soma_nuclei/data/moving/mylsfm_bak/mylsfm_edge.tiff")