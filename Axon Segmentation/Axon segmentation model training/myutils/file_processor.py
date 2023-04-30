import os
from os import listdir
from os.path import join
import SimpleITK as sitk
from skimage.io import imsave
import numpy as np
from PIL import Image
import shutil


def read_tiff_stack(path):
    if os.path.isdir(path):
        images = [np.array(Image.open(join(path, p))) for p in sorted(os.listdir(path))]
        return np.array(images)
    else:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            slice = np.array(img)
            images.append(slice)
        return np.array(images)

def read_tiff_files(root):
    res = []
    for f in sorted(os.listdir(root)):
        res.append(np.array(Image.open(join(root, f))))
    return np.array(res)

def get_dir(path):
    tiffs = [join(path, f) for f in listdir(path) if f[0] != '.']
    return sorted(tiffs)

def rmtree(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def read_nifti(path):
    if os.path.isdir(path):
        images = []
        for p in sorted(os.listdir(path)):
            if ".nii.gz" in p:
                itk_img = sitk.ReadImage(join(path, p))
                img = sitk.GetArrayFromImage(itk_img)
                images.append(img)
            else:
                continue
        return np.array(images)
    else:
        itk_img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(itk_img)
        return np.array(img).astype(np.uint8)

def load_nifti_to_tiff(path, out):
    itk_img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(itk_img)
    imsave(out, img.astype(np.uint16))

def load_tiff_to_nifti(path):
    # if os.path.isdir(path):
    #     images = [np.array(Image.open(os.path.join(path, p))) for p in sorted(os.listdir(path))]
    # else:
    #     img = Image.open(path)
    #     images = []
    #     for i in range():
    #         img.seek(i)
    #         slice = np.array(img)
    #         images.append(slice)
    images = read_tiff_stack(path)
    print(images.shape)
    itk_img = sitk.GetImageFromArray(images)
    return itk_img
