from PIL import Image
import numpy as np
import cv2

def read_tiff_stack(file):
    img = Image.open(file)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)
        images.append(slice)
    return np.array(images)

def resize(data_mask, _x, _y, classes):
    data_mask_new = np.zeros((_y, _x))
    for i in range(classes):
        data_mask1 = np.zeros_like(data_mask)
        data_mask1[data_mask == (i + 1)] = 1
        data_mask1 = cv2.resize(data_mask1, (_x, _y))
        data_mask_new[data_mask1 == 1] = (i + 1)

    # data_mask1 = np.zeros_like(data_mask)
    # data_mask1[data_mask == 1] = 1
    # data_mask1 = cv2.resize(data_mask1, (_x, _y))
    #
    # data_mask2 = np.zeros_like(data_mask)
    # data_mask2[data_mask == 2] = 1
    # data_mask2 = cv2.resize(data_mask2, (_x, _y))
    #
    #
    # data_mask3 = np.zeros_like(data_mask)
    # data_mask3[data_mask == 3] = 1
    # data_mask3 = cv2.resize(data_mask3, (_x, _y))
    #
    # data_mask4 = np.zeros_like(data_mask)
    # data_mask4[data_mask == 4] = 1
    # data_mask4 = cv2.resize(data_mask4, (_x, _y))
    #
    # data_mask_new = np.zeros((data_mask1.shape[0], data_mask1.shape[1]))
    # data_mask_new[data_mask1 == 1] = 1
    # data_mask_new[data_mask2 == 1] = 2
    # data_mask_new[data_mask3 == 1] = 3
    # data_mask_new[data_mask4 == 1] = 4

    return data_mask_new
