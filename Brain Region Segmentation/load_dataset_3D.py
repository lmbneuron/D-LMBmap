import itertools
import os
import cv2
import tifffile
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch
import numpy as np
import SimpleITK as sitk
import config
args = config.args

def equal(im, flg_max):
    flg_min = 0.01
    mi = im.min()
    mx = im.max()
    imax = flg_max * mx + (1 - flg_max) * mi
    imin = flg_min * mx + (1 - flg_min) * mi

    im[im > imax] = imax
    im[im < imin] = imin
    return (im - np.min(im)) / (np.max(im) - np.min(im))

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list

# Train data
class BasicDataset(Dataset):
    def __init__(self, path, name_label, name_unlabel, a):
        self.path = path
        self.img = []
        self.mask = []
        self.a = a
        directoryname = load_file_name_list(os.path.join(path, name_label))
        self.num = len(directoryname)
        undirectoryname = load_file_name_list(os.path.join(path, name_unlabel))
        directoryname.extend(undirectoryname)
        for d in directoryname:  
            name = 'mask_' + args.name
            m = d.replace('tif', 'ome.nii')
            m = m.replace('data', name)

            self.mask.append(os.path.join(path, m))
            self.img.append(os.path.join(path, d))

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        img_name = self.img[i]
        mask_name = self.mask[i]
        data_img = tifffile.imread(img_name)
        new_x, new_y = 0, 0
        if self.a == 'c':
            new_x, new_y = 448, 320       #view1

        if self.a == 'a':
            new_x, new_y = 512, 448     #view2

        dmin = data_img.min()
        dmax = data_img.max()
        data_img = (data_img - dmin) / (dmax - dmin + 1)
        data_img_new = []
        data_img_new.append(data_img)
        data_img_new = np.array(data_img_new)
        data_img = data_img_new.astype(float)
        data_img = torch.from_numpy(data_img)
        if(i<self.num):
            data_mask = sitk.ReadImage(mask_name)
            data_mask = sitk.GetArrayFromImage(data_mask)
            data_mask = 1 * np.asarray(data_mask).astype('uint16')
            cv2.imwrite('temp_m/mask.tiff', data_mask)

            data_mask1 = np.zeros_like(data_mask)
            data_mask1[data_mask == 1] = 1
            data_mask1 = cv2.resize(data_mask1, (new_x, new_y))

            data_mask2 = np.zeros_like(data_mask)
            data_mask2[data_mask == 2] = 1
            data_mask2 = cv2.resize(data_mask2, (new_x, new_y))

            cv2.imwrite('temp_m/mask1.tiff', data_mask1)
            cv2.imwrite('temp_m/mask2.tiff', data_mask2)
            data_mask_new = []
            data_mask_new.append(data_mask1)
            data_mask_new.append(data_mask2)
            data_mask_new = np.array(data_mask_new)
        else:
            data_mask_new = np.zeros((2, new_y, new_x))

        data_mask_new = data_mask_new.astype(float)
        data_mask_new = torch.from_numpy(data_mask_new)

        return {"img": data_img, "cate": data_mask_new, "mask_name": mask_name}

# Test data
class BasicDataset_test(Dataset):
    def __init__(self, path, name):
        self.path = path
        self.img = []
        self.mask = []

        directoryname = load_file_name_list(os.path.join(path, name))
        for d in directoryname:
            self.img.append(os.path.join(path, d))
            self.mask.append(os.path.join(path, d.replace('tif', 'nii')))

    def img_pretreatment(self, img):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.new_W, self.new_H))
        return img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        img_name = self.img[i]
        mask_name = self.mask[i]
        data_img = tifffile.imread(img_name)

        new_x, new_y = 448, 320       #view1
#        new_x, new_y = 512, 448      #view2

        data_img = cv2.resize(data_img, (new_x, new_y))

        dmin = data_img.min()
        dmax = data_img.max()
        if dmin != dmax:
            data_img = (data_img - dmin) / (dmax - dmin)

        data_img = data_img.astype(float)

        data_img = data_img.reshape(1, data_img.shape[0], data_img.shape[1])

        data_img = torch.from_numpy(data_img)

        return {"img": data_img, "mask_name": mask_name}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)