from models import create_model
from multiprocessing import Pool
from options.val_options import ValOptions

import os
import cv2
import sys
import cc3d
import torch
import random
import shutil
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from scipy import misc
from libtiff import TIFF
from einops import rearrange
from scipy.cluster.vq import kmeans,vq,whiten
from skimage.morphology import skeletonize,thin




def read_tiff_stack(file):
    img = Image.open(file)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        slice = np.array(img)
        images.append(slice)
    return np.array(images)


def read_tiff_files(root):
    res = []
    for f in sorted(os.listdir(root)):
        res.append(np.array(Image.open(path(root, f))))
    return np.array(res)


def path(*args):
    return os.path.join(*args)


def rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def equal(im, flg_max):
    flg_min = 0.01
    mi = im.min()
    mx = im.max()
    imax = flg_max * mx + (1 - flg_max) * mi
    imin = flg_min * mx + (1 - flg_min) * mi

    im[im > imax] = imax
    im[im < imin] = imin
    return (im - np.min(im))/(np.max(im) - np.min(im))


def save_resized_tiff_by_scale(root, target, scales, mod='single'):
    images = []
    scale_x, scale_y, scale_z = scales
    if mod == 'single':
        stack = TIFF.open(root, mode='r')
        _y, _x = np.array(list(stack.iter_images())[0]).shape
        for img in list(stack.iter_images()):
            img = np.array(img)
            img = cv2.resize(img, (int(_x * scale_x), int(_y * scale_y)))
            images.append(img)
    elif mod == 'files':
        _y, _x = np.array(Image.open(os.path.join(root, sorted(os.listdir(root))[0]))).shape
        for i in tqdm(sorted(os.listdir(root))):
            img = Image.open(os.path.join(root, i))
            img = np.array(img)
            img = cv2.resize(img, (int(_x * scale_x), int(_y * scale_y)))
            #             tifffile.imsave(root2 + f'/{i}.tiff', img.transpose((1,0)))
            images.append(img)
    images = np.array(images).transpose((2, 1, 0))
    _y, _z = images[0].shape
    res_img = np.array([cv2.resize(im, (int(_z * scale_z), _y)) for im in images]).astype(np.uint8)
    tifffile.imsave(target, res_img.transpose((2, 1, 0)))


def save_resized_tiff_by_shape(root, target, shapes, mod='single'):
    images = []
    shape_x, shape_y, shape_z = shapes
    if mod == 'single':
        stack = TIFF.open(root, mode='r')
        for img in tqdm(list(stack.iter_images())):
            img = np.array(img).astype(np.uint16) / 65535 * 255
            img = cv2.resize(img, (shape_x, shape_y))
            images.append(img)
    elif mod == 'files':
        for i in tqdm(sorted(os.listdir(root))):
            img = Image.open(os.path.join(root, i))
            img = np.array(img)
            img = np.array(img).astype(np.uint16) / 65535 * 255
            img = cv2.resize(img, (shape_x, shape_y))
            #             tifffile.imsave(root + f'/{i}.tiff', img.transpose((1,0)))
            images.append(img)
    images = np.array(images).transpose((2, 1, 0))
    _y, _z = images[0].shape
    res_img = np.array([cv2.resize(im, (shape_z, _y)) for im in images]).astype(np.uint8)
    tifffile.imsave(target, res_img.transpose((2, 1, 0)))


def save_resized_tifs_by_shape(root, target, shapes, mod='single'):
    images = []
    shape_x, shape_y = shapes
    if mod == 'single':
        stack = TIFF.open(root, mode='r')
        for img in tqdm(list(stack.iter_images())):
            img = np.array(img).astype(np.uint8)
            img = cv2.resize(img, (shape_x, shape_y))
            tifffile.imsave(target + i, img)
    elif mod == 'files':
        for i in tqdm(sorted(os.listdir(root))[800:1200]):
            img = Image.open(os.path.join(root, i))
            img = np.array(img)
            img = cv2.resize(img, (shape_x, shape_y))
            #             tifffile.imsave(root2 + f'/{i}.tiff', img.transpose((1,0)))
            tifffile.imsave(target + i, img)


def resize_3d(stack, scales):
    scale_x, scale_y, scale_z = scales
    _z, _y, _x = stack.shape
    images = np.array([cv2.resize(img, (int(_x * scale_x), int(_y * scale_y))) for img in stack])
    images = images.transpose((2, 1, 0))
    _y, _z = images[0].shape
    res_img = np.array([cv2.resize(im, (int(_z * scale_z), _y)) for im in images]).astype(np.uint8)
    return res_img.transpose((2, 1, 0))

if __name__ == '__main__':
    # root = "/media/root/lzy_data/647_Stanford/190911_488_0-8x/"
    # # root = '/media/root/6701ae9d-6612-4271-8d50-522d7f72528b/szy/projects/Sunmap-master/segmentations/dn163751-1/'
    # save_resized_tiff_by_shape(root=root,
    #                            target='../dn_ori.tiff',
    #                            shapes=[500, 500, 500],
    #                            mod='files')
    opt = ValOptions().parse()
    model = create_model(opt)
    val_type = opt.val_type
    if val_type == 'volumes2':
        model.eval_two_volumes_maxpool()
    elif val_type == 'cubes':
        model.eval_volumes_batch()
    elif val_type == 'segment':
        imgs = model.test_3D_volume()
        # pool = Pool(processes=model.opt.process)
        # pool.map(model.segment_brain_batch, imgs)
        # pool.close()
        # pool.join()
        for i in imgs:
            model.segment_brain_batch(i)


